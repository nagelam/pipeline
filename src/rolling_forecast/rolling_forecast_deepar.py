import numpy as np
import pandas as pd
import logging
import traceback
import gc
from typing import Tuple, List
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import MultivariateNormalDistributionLoss, MultiLoss
import lightning.pytorch as pl
from src.models.deepar_model import DeepARForecaster
from lightning.pytorch.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.ERROR)


class RollingForecasterDeepAR:
    """DeepAR модель для совместного прогноза двух временных рядов"""

    def __init__(self, config: dict):
        self.config = config
        self.training_config = config['training']
        self.feature_config = config['feature_engineering']
        self.output_dir = Path(self.config['output']['results_dir'])

    def rolling_forecast(
            self,
            df_with_lags: pd.DataFrame,
            req_col: str,
            err_col: str, layers_config: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Обучает модель по окнам и возвращает агрегированные и дневные метрики

        Args:
            df_with_lags: датафрейм с признаками и таргетами, должен быть отсортирован по времени.
            req_col: Имя первого таргета
            err_col: Имя второго таргета

            Returns:
        (df, df) с агрегированными метриками по всему времени и дневными метриками
        """
        logger.info(f"rolling forecast DeepAR for [{req_col}, {err_col}]")

        timesteps = self.training_config['timesteps']
        train_window = self.training_config['train_window']
        predict_window = self.training_config['predict_window']
        epochs = self.training_config['epochs']
        lr = self.training_config['learning_rate']
        batch_size = self.training_config['batch_size']

        """
                cfg_req = self.feature_config['requests_lags']
                cfg_err = self.feature_config['errors_lags']
                lags_req = []
                for key, values in cfg_req.items():
                    lags_req.extend(values)
                lags_req = sorted({int(l) for l in lags_req if isinstance(l, int) and l >= 0 and l <timesteps//2})

                lags_err = []
                for key, values in cfg_req.items():
                    lags_err.extend(values)
                lags_err = sorted({int(l) for l in lags_err if isinstance(l, int) and l >= 0 and l < timesteps//2})
        """
        # разделение на признаки и таргеты
        y_cols = [req_col, err_col]
        feature_cols = [c for c in df_with_lags.columns if c not in y_cols]
        req_err_lag = [c for c in feature_cols if c.startswith(('req', 'err'))]  # лаги
        feature = [c for c in feature_cols if c not in req_err_lag]  # значения как час, неделя, синус_час
        X_df = df_with_lags.drop(columns=y_cols)
        Y_df = df_with_lags[y_cols]

        # скалирование
        x_scaler = MinMaxScaler().fit(X_df)
        y_scaler = MinMaxScaler().fit(Y_df)
        Xs = pd.DataFrame(x_scaler.transform(X_df), index=X_df.index, columns=X_df.columns)
        Ys = Y_df

        all_preds = []
        mae_req_list, mae_err_list, mse_req_list, mse_err_list = [], [], [], []
        start = 0
        window_id = 0
        total = len(df_with_lags)

        while start + train_window < total:
            # сколько осталось
            avail = total - (start + train_window)
            # минимум из того сколько осталось, и на сколько мы хотим предсказывать
            curr_pred = min(predict_window, avail)

            if curr_pred < timesteps + 1:
                break

            # тренировочные данные по train_window
            X_train_df = Xs.iloc[start:start + train_window].reset_index(drop=True)
            y_train_df = Ys.iloc[start:start + train_window].reset_index(drop=True)

            # тестовые данные по curr_pred
            X_test_df = Xs.iloc[start + train_window: start + train_window + curr_pred]
            y_test_df = Ys.iloc[start + train_window: start + train_window + curr_pred]

            if len(X_train_df) < timesteps + 1:
                break

            logger.info(f"DeepAR Window {window_id}: train={len(X_train_df)}, test={len(X_test_df)}")

            try:
                train_index = df_with_lags.index[start:start + train_window]
                train_data = pd.DataFrame(
                    np.hstack([X_train_df.values, y_train_df.values]),
                    index=train_index,
                    columns=list(X_df.columns) + y_cols
                )
                train_data["group"] = 0
                train_data["time_idx"] = np.arange(len(train_data))

                train_ds = TimeSeriesDataSet(
                    train_data,
                    time_idx="time_idx",
                    target=y_cols,
                    group_ids=["group"],
                    min_encoder_length=max(1, timesteps // 2),
                    max_encoder_length=timesteps,
                    min_prediction_length=1,
                    max_prediction_length=1,
                    time_varying_known_reals=feature + req_err_lag,
                    time_varying_unknown_reals=y_cols,
                    target_normalizer=MultiNormalizer([
                        GroupNormalizer(groups=["group"], method="standard"),
                        GroupNormalizer(groups=["group"], method="standard")
                    ]),
                    add_relative_time_idx=True,
                    add_encoder_length=True,
                    # lags={req_col:lags_req,err_col:lags_req}

                )

                model = DeepARForecaster.build_model(
                    self,
                    layers_config=layers_config,
                    dataset=train_ds,
                    learning_rate=lr
                )
                trainer = pl.Trainer(
                    max_epochs=epochs,
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    logger=False,
                    enable_progress_bar=False
                )
                trainer.fit(model, train_ds.to_dataloader(train=True, batch_size=batch_size))

                feat_stream = X_test_df.values
                test_index = X_test_df.index

                init_input = feat_stream[:timesteps][None]
                preds_scaled = []

                for i in range(len(feat_stream) - timesteps + 1):
                    window_df = pd.DataFrame(
                        init_input[0],
                        columns=X_df.columns
                    )
                    for t in y_cols:
                        window_df[t] = 0.0
                    window_df["group"] = 0
                    window_df["time_idx"] = np.arange(timesteps)

                    ts_ds = TimeSeriesDataSet.from_dataset(
                        train_ds,
                        window_df,
                        predict=True,
                        stop_randomization=True,  # без случайной маскировки для воспроизводимости
                    )
                    dl = ts_ds.to_dataloader(train=False, batch_size=1)
                    p_vec = model.predict(dl)

                    p_vec = torch.cat(p_vec, dim=1).squeeze(0).cpu()
                    preds_scaled.append(p_vec.numpy())
                    if i + timesteps < len(feat_stream):
                        next_feat = feat_stream[i + timesteps][None]
                        window = np.vstack([init_input[0, 1:], next_feat])
                        init_input = window[None]

                preds_arr = np.asarray(preds_scaled)
                preds = y_scaler.inverse_transform(preds_arr)

                trues_s = y_test_df.values[timesteps - 1:]
                trues = y_scaler.inverse_transform(trues_s)
                dates = y_test_df.index[timesteps - 1:]
                # pd.DataFrame(dates).to_csv('deepar(window_id).csv}', mode="a")
                for i, dt in enumerate(dates):
                    all_preds.append({
                        "date": dt,
                        "true_req": trues[i, 0], "pred_req": preds[i, 0],
                        "true_err": trues[i, 1], "pred_err": preds[i, 1]
                    })

                # метрики для окна
                mae_req = mean_absolute_error(trues[:, 0], preds[:, 0])
                mae_err = mean_absolute_error(trues[:, 1], preds[:, 1])
                mse_req = mean_squared_error(trues[:, 0], preds[:, 0])
                mse_err = mean_squared_error(trues[:, 1], preds[:, 1])

                mae_req_list.append(mae_req)
                mae_err_list.append(mae_err)
                mse_req_list.append(mse_req)
                mse_err_list.append(mse_err)

                logger.info(f"Window {window_id} completed: MAE_req={mae_req:.4f}, MAE_err={mae_err:.4f}")

            except Exception as e:
                logger.error(f"Error in window {window_id}: {e}")
                traceback.print_exc()

            start += max(timesteps, curr_pred - timesteps + 1)
            window_id += 1
            gc.collect()

        agg_metrics = pd.DataFrame({
            "MAE_req_mean": [np.mean(mae_req_list)] if mae_req_list else [np.nan],
            "MAE_err_mean": [np.mean(mae_err_list)] if mae_err_list else [np.nan],
            "MSE_req_mean": [np.mean(mse_req_list)] if mse_req_list else [np.nan],
            "MSE_err_mean": [np.mean(mse_err_list)] if mse_err_list else [np.nan]
        })

        # дневные метрики
        if all_preds:
            preds_df = pd.DataFrame(all_preds)
            preds_df["day"] = preds_df["date"].dt.date
            daily_metrics = preds_df.groupby("day").apply(
                lambda g: pd.Series({
                    "MAE_req": mean_absolute_error(g["true_req"], g["pred_req"]),
                    "MAE_err": mean_absolute_error(g["true_err"], g["pred_err"]),
                    "MSE_req": mean_squared_error(g["true_req"], g["pred_req"]),
                    "MSE_err": mean_squared_error(g["true_err"], g["pred_err"])
                })
            ).reset_index()
        else:
            daily_metrics = pd.DataFrame()

        return agg_metrics, daily_metrics
