import numpy as np
import pandas as pd
import logging
import gc
import torch
from typing import Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from pytorch_forecasting import TimeSeriesDataSet
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch import Trainer

from src.models.mlp_model import DecoderMLPForecaster

import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


class RollingForecasterMLP:
    """
    Скользящее окно  DecoderMLP для одного таргета.
    """

    def __init__(self, config: dict):
        self.config = config
        self.training_config = config['training']
        self.forecaster = DecoderMLPForecaster(config)
        self.output_dir = Path(self.config['output']['results_dir'])

    def rolling_forecast(
            self,
            df_with_lags: pd.DataFrame,
            target_col: str, second_target: str,
            layers_config: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Обучает DecoderMLP по скользящим окнам и возвращает агрегированные и дневные метрики.

        Args:
            df_with_lags: датафрейм с таргетом
            target_col: название таргета
            second_target: таргет колонка для другого датасета
            layers_config: список конфигураций слоёв decoder_mlp

        Returns:
            agg_metrics: df с агрегированными метриками по всем окнам
            daily_metrics: df с метриками по дням.
        """

        logger.info(f"Starting rolling forecast DecoderMLP for {target_col}")

        max_encoder_length = self.training_config['max_encoder_length']
        train_window = self.training_config['train_window']
        predict_window = self.training_config['predict_window']
        epochs = self.training_config['epochs']
        lr = self.training_config['learning_rate']
        batch_size = self.training_config['batch_size']

        # Разделяем признаки и таргет
        y_cols = [target_col, second_target]
        feature_cols = [c for c in df_with_lags.columns if c not in y_cols]
        req_err_lag = [c for c in feature_cols if c.startswith(('req', 'err'))]  # лаги
        feature = [c for c in feature_cols if c not in req_err_lag]  # значения как час, неделя, синус_час

        scaler_feat = MinMaxScaler().fit(df_with_lags[feature_cols])
        scaler_tgt = MinMaxScaler().fit(df_with_lags[[target_col]])

        scaled = pd.DataFrame(
            scaler_feat.transform(df_with_lags[feature_cols]),
            columns=feature_cols,
            index=df_with_lags.index,
        )
        scaled[target_col] = scaler_tgt.transform(df_with_lags[[target_col]])

        total = len(scaled)
        mae_list, mse_list = [], []
        all_preds = []
        start = 0
        window_id = 0

        while start + train_window < total:
            # сколько осталось точек
            avail = total - (start + train_window)
            # минимум из желаемого горизонта и того что осталось
            curr_pred = min(predict_window, avail)
            if curr_pred < max_encoder_length + 1:
                break
            # тут размеры по train_window
            train_df = scaled.iloc[start: start + train_window].copy()
            # тут размеры по curr_pred
            test_df = scaled.iloc[start + train_window: start + train_window + curr_pred].copy()

            if len(train_df) < max_encoder_length + 1:
                break

            logger.info(f"DecoderMLP Window {window_id}: train={len(train_df)}, test={len(test_df)}")

            full = train_df.reset_index(drop=True).copy()
            full["time_idx"] = np.arange(len(full))
            full["series"] = "0"

            val_size = max(1, int(len(full) * 0.1))
            train_full = full.iloc[:-val_size].copy()
            val_full = full.iloc[-val_size:].copy()

            train_ds = TimeSeriesDataSet(
                train_full,
                time_idx="time_idx",
                target=target_col,
                group_ids=["series"],
                max_encoder_length=max_encoder_length,
                max_prediction_length=1,
                time_varying_unknown_reals=[target_col] + req_err_lag,
                time_varying_known_reals=feature,
                static_categoricals=["series"],
            )

            val_ds = TimeSeriesDataSet.from_dataset(
                train_ds, val_full, predict=False, stop_randomization=True
            )

            train_dl = train_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
            val_dl = val_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

            model = self.forecaster.build_model(layers_config, max_encoder_length, train_ds, lr)

            callbacks = [EarlyStopping(monitor="val_loss", patience=self.training_config['early_stopping_patience'])]

            if torch.cuda.is_available():
                trainer = pl.Trainer(
                    max_epochs=epochs,
                    accelerator="gpu",
                    callbacks=callbacks,
                    enable_progress_bar=False,
                    logger=False
                )
            else:
                trainer = pl.Trainer(
                    max_epochs=epochs,
                    accelerator="cpu",
                    callbacks=callbacks,
                    enable_progress_bar=False,
                    logger=False
                )

            trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

            # Инференс
            full_seq = pd.concat([train_df, test_df]).reset_index(drop=True).copy()
            full_seq["time_idx"] = np.arange(len(full_seq))
            full_seq["series"] = "0"

            infer_ds = TimeSeriesDataSet(
                full_seq,
                time_idx="time_idx",
                target=target_col,
                group_ids=["series"],
                max_encoder_length=max_encoder_length,
                max_prediction_length=1,
                time_varying_unknown_reals=[target_col] + req_err_lag,
                time_varying_known_reals=feature,
                static_categoricals=["series"],
                allow_missing_timesteps=True,
            )

            infer_dl = infer_ds.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

            pred_all = model.predict(infer_dl).cpu().numpy().flatten()

            if len(pred_all) >= len(test_df):
                preds_scaled = pred_all[-len(test_df):].reshape(-1, 1)
            else:
                preds_scaled = pred_all[-min(len(pred_all), len(test_df)):].reshape(-1, 1)

            preds = scaler_tgt.inverse_transform(preds_scaled).flatten()

            preds = np.maximum(preds, 0)

            trues = scaler_tgt.inverse_transform(test_df[target_col].values[:len(preds)].reshape(-1, 1)).flatten()

            mae = mean_absolute_error(trues, preds)
            mse = mean_squared_error(trues, preds)
            mae_list.append(mae)
            mse_list.append(mse)

            dates = test_df.index[:len(preds)]
            # pd.DataFrame(dates).to_csv('mlp{window_id}.csv', mode="a")
            for ts, t, pr in zip(dates, trues, preds):
                all_preds.append({"date": ts, "true": t, "pred": pr})

            if self.config['output']['save_results']:
                preds_df = pd.DataFrame(all_preds)
                preds_df['date'] = pd.to_datetime(preds_df['date'])

                true_full = df_with_lags[[target_col]].copy()
                true_full.index = pd.to_datetime(true_full.index)

                plt.figure(figsize=(12, 6))
                plt.plot(true_full.index.values, true_full[target_col].values, label='True (full)', color='steelblue')

                # plt.figure(figsize=(12, 6))
                plt.plot(preds_df['date'].values, preds_df['true'].values, label='True', color='blue')
                plt.plot(preds_df['date'].values, preds_df['pred'].values, label='Pred', color='orange', linestyle='--')
                plt.xlabel('Date')
                plt.ylabel('Value')
                plt.title(f'True vs Pred {target_col}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                plot_path = self.output_dir / f'{target_col}, decoder_mlp.png'
                plt.savefig(plot_path)
                plt.close()

            logger.info(f"DecoderMLP Window {window_id}: MAE={mae:.3f}, MSE={mse:.3f}")

            start += curr_pred
            window_id += 1
            gc.collect()

        # Агрегированные метрики
        agg = pd.DataFrame({
            "MAE_mean": [np.mean(mae_list) if mae_list else np.nan],
            "MAE_median": [np.median(mae_list) if mae_list else np.nan],
            "MSE_mean": [np.mean(mse_list) if mse_list else np.nan],
            "MSE_median": [np.median(mse_list) if mse_list else np.nan],
        })

        # Дневные метрики
        preds_df = pd.DataFrame(all_preds)
        if not preds_df.empty:
            preds_df["day"] = pd.to_datetime(preds_df["date"]).dt.date
            daily = (
                preds_df.groupby("day")
                .apply(lambda g: pd.Series({
                    "MAE": mean_absolute_error(g["true"], g["pred"]),
                    "MSE": mean_squared_error(g["true"], g["pred"]),
                }))
                .reset_index()
            )
        else:
            daily = pd.DataFrame(columns=["day", "MAE", "MSE"])

        return agg, daily

    def rolling_forecast_single(
            self,
            df_with_lags: pd.DataFrame,
            target_col: str,
            layers_config: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        скользящие окна для одного таргета используя основной метод
        """
        return self.rolling_forecast(df_with_lags, target_col, "", layers_config)
