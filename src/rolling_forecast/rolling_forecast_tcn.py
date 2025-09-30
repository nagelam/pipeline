import numpy as np
import pandas as pd
import logging
import traceback
import gc
from typing import Tuple

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from src.models.tcn_model import TCNForecaster

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
from pathlib import Path


class RollingForecasterTCN:
    """TCN модель для совместного прогноза двух временных рядов"""

    def __init__(self, config: dict):
        self.config = config
        self.training_config = config['training']
        self.forecaster = TCNForecaster(config)
        self.output_dir = Path(self.config['output']['results_dir'])

    def rolling_forecast(
            self,
            df_with_lags: pd.DataFrame,
            req_col: str,
            err_col: str,
            layers_config: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Обучает TCN по окнам и возвращает агрегированные и дневные метрики.

        Args:
            df_with_lags: датафрейм с признаками и таргетоми, должен быть отсортированн по времени.
            req_col: Имя первого таргета
            err_col: Имя второго таргета
            layers_config: Список конфига TCN
        Returns:
            (df, df) с агрегированными метриками по всему времени и дневными метриками
        """
        logger.info(f"Starting rolling forecast TCN for {req_col} {err_col}")

        timesteps = self.training_config['timesteps']
        train_window = self.training_config['train_window']
        predict_window = self.training_config['predict_window']
        epochs = self.training_config['epochs']
        lr = self.training_config['learning_rate']
        batch_size = self.training_config['batch_size']

        # Разделяем признаки и таргеты
        y_cols = [req_col, err_col]
        X_df = df_with_lags.drop(columns=y_cols, errors='ignore')
        Y_df = df_with_lags[y_cols].copy()

        # скалирование
        x_scaler = MinMaxScaler().fit(X_df.values)
        y_scaler = MinMaxScaler().fit(Y_df.values)  # общий скейлер для 2 таргетов

        Xs = pd.DataFrame(
            x_scaler.transform(X_df.values),
            columns=X_df.columns,
            index=df_with_lags.index
        )
        Ys = pd.DataFrame(
            y_scaler.transform(Y_df.values),
            columns=y_cols,
            index=df_with_lags.index
        )

        total = len(df_with_lags)
        mae_req_list, mae_err_list, mse_req_list, mse_err_list = [], [], [], []
        all_preds = []
        start = 0
        window_id = 0

        while start + train_window < total:
            # сколько осталось
            avail = total - (start + train_window)
            # минимум из того сколько осталось, и на сколько мы хотим предсказывать
            curr_pred = min(predict_window, avail)
            if curr_pred < timesteps + 1:
                break
            # тут размеры по train_window
            X_train_df = Xs.iloc[start:start + train_window].reset_index(drop=True)
            y_train_df = Ys.iloc[start:start + train_window].reset_index(drop=True)
            # тут размеры по curr_pred
            X_test_df = Xs.iloc[start + train_window: start + train_window + curr_pred]
            y_test_df = Ys.iloc[start + train_window: start + train_window + curr_pred]

            if len(X_train_df) < timesteps + 1:
                break

            logger.info(f"TCN Window {window_id}: train={len(X_train_df)}, test={len(X_test_df)}")

            try:
                # тут перекрывающиеся окна для тренировки
                X_train, Y_train = self.create_sequences(
                    X_train_df.values,
                    y_train_df.values,
                    timesteps
                )

                model = self.forecaster.build_model(
                    layers_config, timesteps, X_train.shape[2], lr, n_outputs=2
                )

                es = EarlyStopping(
                    monitor='val_loss',
                    patience=self.training_config['early_stopping_patience'],
                    restore_best_weights=True
                )

                model.fit(
                    X_train, Y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=self.training_config['validation_split'],
                    callbacks=[es],
                    verbose=0
                )

                # Пошчасовой прогноз на тесте
                feat_stream = X_test_df.values

                init_input = feat_stream[:timesteps][np.newaxis, ...]  # (1, длина таймстепс, число колонок)
                preds_scaled = []
                for i in range(len(feat_stream) - timesteps + 1):
                    p_vec = model.predict(init_input, verbose=0)  # (1,2)
                    preds_scaled.append(p_vec)

                    # Сдвигаем окно
                    if i + timesteps < len(feat_stream):
                        next_feat = feat_stream[i + timesteps].reshape(1, -1)
                        window = np.vstack(
                            [init_input[0, 1:], next_feat])  # (длина таймстепс, число колонок) сдвиг на 1

                        init_input = window[np.newaxis, ...]  # обратно к (1, длина таймстепс, число колонок)
                preds_arr = np.asarray(preds_scaled)  # (число предсказаний, 1, 2)

                preds_arr = preds_arr[:, -1, :]  # (число предсказаний, 2)

                preds = y_scaler.inverse_transform(preds_arr)
                preds = np.maximum(preds, 0)

                trues_s = y_test_df.values[timesteps - 1:]  # (число предсказаний, 2)
                trues = y_scaler.inverse_transform(trues_s)  # (число предсказаний, 2)

                dates = pd.to_datetime(y_test_df.index)[timesteps - 1:]
                # pd.DataFrame(dates).to_csv('tcn(window_id).csv', mode="a")

                # Метрики по окну
                mae_req = mean_absolute_error(trues[:, 0], preds[:, 0])
                mae_err = mean_absolute_error(trues[:, 1], preds[:, 1])
                mse_req = mean_squared_error(trues[:, 0], preds[:, 0])
                mse_err = mean_squared_error(trues[:, 1], preds[:, 1])

                mae_req_list.append(mae_req)
                mae_err_list.append(mae_err)
                mse_req_list.append(mse_req)
                mse_err_list.append(mse_err)

                for ts, (t_req, t_err), (p_req, p_err) in zip(dates, trues, preds):
                    dt_day_hour = pd.to_datetime(ts).replace(minute=0, second=0, microsecond=0)
                    all_preds.append({'date': dt_day_hour,
                                      'true_req': t_req, 'pred_req': p_req,
                                      'true_err': t_err, 'pred_err': p_err})

                logger.info(f"TCN-2out Window {window_id}: "
                            f"MAE(req)={mae_req:.3f}, MAE(err)={mae_err:.3f}, "
                            f"MSE(req)={mse_req:.3f}, MSE(err)={mse_err:.3f}")
                start += max(timesteps, curr_pred - timesteps + 1)

            except Exception as e:
                logger.error(f"TCN-2out Error in window {window_id}: {e}")
                logger.debug(traceback.format_exc())
                traceback.print_exc()

            finally:
                window_id += 1
                if 'model' in locals():
                    self.cleanup_model(model)

        # Агрегированные метрики по окнам
        agg_metrics = pd.DataFrame({
            'MAE_req_mean': [np.mean(mae_req_list)],
            'MAE_req_median': [np.median(mae_req_list)],
            'MAE_err_mean': [np.mean(mae_err_list)],
            'MAE_err_median': [np.median(mae_err_list)],
            'MSE_req_mean': [np.mean(mse_req_list)],
            'MSE_req_median': [np.median(mse_req_list)],
            'MSE_err_mean': [np.mean(mse_err_list)],
            'MSE_err_median': [np.median(mse_err_list)],
            'MAE_mean_overall': [np.mean(mae_req_list + mae_err_list)],
            'MSE_mean_overall': [np.mean(mse_req_list + mse_err_list)]
        })

        preds_df = pd.DataFrame(all_preds)
        preds_df['day'] = preds_df['date'].dt.date

        if self.config['output']['save_results']:
            preds_df['date'] = pd.to_datetime(preds_df['date'])
            true_full = df_with_lags[[req_col]].copy()
            true_full.index = pd.to_datetime(true_full.index)

            plt.figure(figsize=(12, 6))
            # полная истина по всему промежутку
            plt.plot(true_full.index.values, true_full[req_col].values, label='True (full)', color='steelblue')

            # plt.figure(figsize=(12, 6))
            plt.plot(preds_df['date'].values, preds_df['true_req'].values, label='True Requests', color='blue')
            plt.plot(preds_df['date'].values, preds_df['pred_req'].values, label='Pred Requests', color='orange',
                     linestyle='--')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.title('True vs Pred Requests')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            req_plot_path = self.output_dir / 'req_target, tcn.png'
            plt.savefig(req_plot_path)
            plt.close()
            logger.info(f"Saved requests true vs pred plot to {req_plot_path}")

            true_full = df_with_lags[[err_col]].copy()
            true_full.index = pd.to_datetime(true_full.index)

            plt.figure(figsize=(12, 6))
            plt.plot(true_full.index.values, true_full[err_col].values, label='True (full)', color='steelblue')

            # plt.figure(figsize=(12, 6))
            plt.plot(preds_df['date'].values, preds_df['true_err'].values, label='True Errors', color='blue')
            plt.plot(preds_df['date'].values, preds_df['pred_err'].values, label='Pred Errors', color='orange',
                     linestyle='--')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.title('True vs Pred Errors')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            err_plot_path = self.output_dir / 'err_target, tcn.png'
            plt.savefig(err_plot_path)
            plt.close()
            logger.info(f"Saved errors true vs pred plot to {err_plot_path}")

        # Дневные метрики
        daily_metrics = (
            preds_df.groupby('day')
            .apply(lambda g: pd.Series({
                'MAE_req': mean_absolute_error(g['true_req'], g['pred_req']),
                'MSE_req': mean_squared_error(g['true_req'].values, g['pred_req'].values),
                'MAE_err': mean_absolute_error(g['true_err'], g['pred_err']),
                'MSE_err': mean_squared_error(g['true_err'].values, g['pred_err'].values),
                'MAE_avg': 0.5 * (mean_absolute_error(g['true_req'], g['pred_req']) +
                                  mean_absolute_error(g['true_err'], g['pred_err'])),
                'MSE_avg': 0.5 * (mean_squared_error(g['true_req'].values, g['pred_req'].values) +
                                  mean_squared_error(g['true_err'].values, g['pred_err'].values))
            }))
            .reset_index()
        )

        return agg_metrics, daily_metrics

    def create_sequences(self, data: np.ndarray, target_df: np.ndarray, timesteps: int):
        """
        Формирует массивы по таймстепс X, Y для окна, при этом окна идут с шагом 1(и пересекаются)
        Args:
            data: Матрица признаков
            target_df: Матрица таргетов (число строк, 2 таргета)
            timesteps: Длина timesteps

        Returns:
             X — массив срезов дата
             Y — массив формы (число окон, 2)
        """
        X, Y = [], []
        for i in range(len(data) - timesteps):
            X.append(data[i:i + timesteps])
            Y.append(target_df[i + timesteps])
        return np.array(X), np.array(Y)

    def cleanup_model(self, model):
        del model
        K.clear_session()
        gc.collect()
