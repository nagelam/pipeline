import pandas as pd
import numpy as np
import logging
import traceback
import gc
from tensorflow.keras.models import Sequential

import tensorflow.keras.backend as K
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple
from tensorflow.keras.callbacks import EarlyStopping

from src.models.lstm_model import LSTMForecaster
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


class RollingForecaster:
    """
    Скользящее окно прогнозирование при помощи LSTM-модели.

    Идея:
        Делим исходный датафрейм на окна, обучаемся на размере train_window
       и для каждого окна строим локальную LSTM

        Модель предсказывает на размер predict_window(или меньше если данных на n шаге < predict_window)
        по каждому часу

        Метрики MAE и MSE усредняются по всем окнам
        матрики ежедневных предсказаний усредняются по дням
    """

    def __init__(self, config: dict):
        self.config = config
        self.training_config = config['training']
        self.forecaster = LSTMForecaster(config)
        self.output_dir = Path(self.config['output']['results_dir'])

    def rolling_forecast(
            self,
            df_with_lags: pd.DataFrame,
            target_col: str,
            second_target: str,
            layers_config: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        обучает LSTM модель на скользящих окнах
        возвращает усредненные метрики по дням и окнам

        Args:
            df_with_lags: датафрейм с лагами
            target_col: таргет колонка
            second_target: таргет колонка для другого датасета
                        Мы хотим сцепить ряды и использовать их взаимосвязь
                        В момент предсказания запросов мы не можем использовать число ошибок
                        на время которое предсказываем
                        Поэтому мы выкидываем таргет другого датасета
                        При этом лаги другого датасета мы не выкидываем,
                        так как знаем сколько ошибок было вчера или час назад
            layers_cfg: конфигурация LSTM-слоёв в конфиге

        Returns:
            agg_metrics:датафрейм с усредненными метриками MAE MSE по окнам
            daily_metrics: датафрейм с усредненными метриками MAE MSE для дня
        """
        logger.info(f"Starting rolling forecast LSTM for {target_col}")

        timesteps = self.training_config['timesteps']
        train_window = self.training_config['train_window']
        predict_window = self.training_config['predict_window']
        epochs = self.training_config['epochs']
        lr = self.training_config['learning_rate']
        batch_size = self.training_config['batch_size']

        # Убираем второй таргет
        df = df_with_lags.drop(columns=[second_target], errors='ignore')

        # скалируем
        scaler_feat = MinMaxScaler().fit(df.drop(columns=[target_col]))
        scaler_tgt = MinMaxScaler().fit(df[[target_col]])
        scaled = pd.DataFrame(
            scaler_feat.transform(df.drop(columns=[target_col])),
            columns=df.drop(columns=[target_col]).columns,
            index=df.index
        )
        scaled[target_col] = scaler_tgt.transform(df[[target_col]])

        total = len(scaled)
        mae_list, mse_list = [], []
        all_preds = []
        start = 0
        window_id = 0
        # цикл по окнам
        while start + train_window < total:
            avail = total - (start + train_window)
            # длина тестового сегмента
            curr_pred = min(predict_window, avail)
            if curr_pred < timesteps + 1:
                break
            # обучающий и тестовый срез
            train_df = scaled.iloc[start:start + train_window].reset_index(drop=True)
            test_df = scaled.iloc[start + train_window:
                                  start + train_window + curr_pred]
            if len(train_df) < timesteps + 1:
                break

            logger.info(f"Window {window_id}: train={len(train_df)}, test={len(test_df)}")

            try:
                # из Х убираем оба таргета, добавляем в у нужный
                y_cols = [target_col]
                X_train, y_train = self.create_sequences(
                    train_df.drop(columns=y_cols).values,
                    train_df[target_col].values,
                    timesteps
                )

                model = self.forecaster.build_model(
                    layers_config, timesteps, X_train.shape[2], lr
                )
                es = EarlyStopping(
                    monitor='val_loss',
                    patience=self.training_config['early_stopping_patience'],
                    restore_best_weights=True
                )
                model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[es],
                    verbose=0
                )

                # прогнозирование по часу
                test_feats = test_df.drop(columns=[target_col]).values

                init_input = test_feats[:timesteps][np.newaxis, ...]  # (1, длина таймстепс, число колонок)

                preds_scaled = []
                for i in range(len(test_feats) - timesteps + 1):
                    p = model.predict(init_input, verbose=0)[0, 0]
                    preds_scaled.append(p)
                    # Сдвигаем окно
                    if i + timesteps < len(test_feats):
                        next_feat = test_feats[i + timesteps: i + timesteps + 1]  # (1, число столбцов)

                        init_input = np.vstack([init_input[0, 1:], next_feat])[
                            np.newaxis, ...]  # (1, длина таймстепс, число колонок)

                # обратное масштабирование
                preds = scaler_tgt.inverse_transform(
                    np.array(preds_scaled).reshape(-1, 1)
                ).flatten()

                preds = np.maximum(preds, 0)

                trues = scaler_tgt.inverse_transform(
                    test_df[target_col].values[timesteps - 1:].reshape(-1, 1)
                ).flatten()
                dates = pd.to_datetime(test_df.index)[timesteps - 1:]
                # pd.DataFrame(dates).to_csv('lstm(window_id).csv',mode="a")
                # метрики по текущему окну которые потом будут усредненны по дням и поокнам
                mae = mean_absolute_error(trues, preds)
                sm = mean_squared_error(trues, preds)
                # добавляем в массив для вычисление средних метрик по окнам
                mae_list.append(mae)
                mse_list.append(sm)

                for ts, t, pr in zip(dates, trues, preds):
                    dt_day_hour = pd.to_datetime(ts).replace(minute=0, second=0, microsecond=0)

                    all_preds.append({'date': dt_day_hour, 'true': t, 'pred': pr})

                logger.info(f"Window {window_id}: MAE={mae:.3f}, MSE={sm:.3f}")
                # сдвигаем окно
                start += max(timesteps, curr_pred - timesteps + 1)


            except Exception as e:
                logger.error(f"Error in window {window_id}: {e}")

            finally:
                window_id += 1
                if 'model' in locals():
                    self.cleanup_model(model)

        agg_metrics = pd.DataFrame({
            'MAE_mean': [np.mean(mae_list)],
            'MAE_median': [np.median(mae_list)],
            'MSE_mean': [np.mean(mse_list)],
            'MSE_median': [np.median(mse_list)]
        })

        preds_df = pd.DataFrame(all_preds)
        preds_df['day'] = preds_df['date'].dt.date
        if self.config['output']['save_results']:
            true_full = df[[target_col]].copy()
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
            plot_path = self.output_dir / f'{target_col}, lstm.png'
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"true vs pred {plot_path}")

        daily_metrics = (
            preds_df.groupby('day')
            .apply(lambda g: pd.Series({
                'MAE': mean_absolute_error(g['true'], g['pred']),
                'MSE': mean_squared_error(g['true'].values, g['pred'].values),
            }))
            .reset_index()
        )

        return agg_metrics, daily_metrics

    def create_sequences(self, data: np.ndarray, target: np.ndarray, timesteps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Преобразует данные в последовательсности для LSTM , при этом окна идут с шагом 1

        Args:
            data: Массив признаков
            target: массив таргета
            timesteps: Длина окна контекста для LSTM

        Returns:
            X: масив срезов data
            y: массив срезов target
        """
        X, y = [], []
        for i in range(len(data) - timesteps):
            X.append(data[i:i + timesteps])
            y.append(target[i + timesteps])
        return np.array(X), np.array(y)

    def cleanup_model(self, model: Sequential) -> None:
        del model
        K.clear_session()
        gc.collect()

    def rolling_forecast_single(
            self,
            df_with_lags: pd.DataFrame,
            target_col: str,
            layers_config: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Предсказание для одного ряда без сцепки с другим"""
        return self.rolling_forecast(df_with_lags, target_col, 'none', layers_config)
