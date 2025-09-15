import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Работа с признаками для временного ряда"""

    def __init__(self, config: dict):
        """
        Инициализация класса для работы с признаками.
        
        Args:
            config: Словарь параметров
        """
        self.config = config
        self.feature_config = config['feature_engineering']
        self.scaler_features = None
        self.scaler_target = None
        self.target_col = None

    def extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлекает признаки на основе даты и времени из датафрейма.
        
        Args:
            df: датафрейм с колонкой datetime
            
        Returns:
            датафрейм с добавленными признаками 
        """

        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        earliest = df['datetime'].min()
        df['year'] = df['datetime'].dt.year
        df['datediff_in_days'] = (df['datetime'] - earliest).dt.days
        # min max для sin cos преобразований
        time_features = {
            'hour': [0, 23],
            'dayofweek': [0, 6],
            'week': [1, 52],
            'month': [1, 12]
        }
        for col, (cmin, cmax) in time_features.items():
            if col == 'week':
                df[col] = df['datetime'].dt.isocalendar().week.astype(int)
            else:
                df[col] = getattr(df['datetime'].dt, col)
            angles = 2 * np.pi * (df[col] - cmin) / (cmax - cmin + 1)
            df[f'{col}_sin'] = np.sin(angles)
            df[f'{col}_cos'] = np.cos(angles)
        logger.info(f"Added datetime features: "
                    f"{[c for c in df.columns if c.endswith(('_sin', '_cos', 'year', 'datediff_in_days'))]}")
        return df

    def add_lags(self, df: pd.DataFrame, type_df: str) -> pd.DataFrame:
        """
        Добавляет лаги к таргету в датафрейме.
        Args:
            df: датафрейм с таргетом колонкой.
            type_df: Вид датафреймп (req для requests или другой для errors).
            
        Returns:
            датафрейм с добавленными колонками лагов.
        """
        # в зависимости от датафрейма нам нужно по разному делать лаги
        if type_df == 'req':
            cfg = self.feature_config['requests_lags']
            target_col = self.config['data']['renamed_requests_col']
        else:
            cfg = self.feature_config['errors_lags']
            target_col = self.config['data']['renamed_errors_col']
        # грузим лаги из конфига и проверяем их
        lags = []
        for key, values in cfg.items():
            lags.extend(values)
        lags = sorted({int(l) for l in lags if isinstance(l, int) and l >= 0})
        if not lags:
            raise ValueError("No valid lags")
        # добавляем лаги
        df_l = df.copy()
        for lag in lags:
            df_l[f"{target_col}_lag_{lag}"] = df_l[target_col].shift(lag)
        df_l.dropna(inplace=True)
        df_l.reset_index(drop=True, inplace=True)
        logger.info(f"Added {len(lags)} lags for {target_col}")
        return df_l

    def merge_dataframes(self, req: pd.DataFrame, err: pd.DataFrame) -> pd.DataFrame:
        """
        Объединяет датафрейм requests и errors по колонке datetime.
        
        Args:
            req: датафрейм requests.
            err: датафрейм errors.
            
        Returns:
            Объединённый датафрейм.
        """
        # берем из error таргет и лаги
        err_col = self.config['data']['renamed_errors_col']
        lag_cols = [c for c in err.columns if c.startswith(f"{err_col}_lag_")]
        cols = ['datetime', err_col] + lag_cols
        dfm = req.merge(err[cols], on='datetime', how='left')
        # добавляем таргет error на 2 позицию
        if err_col in dfm.columns:
            cols = dfm.columns.tolist()
            cols.insert(1, cols.pop(cols.index(err_col)))
            dfm = dfm[cols]
        logger.info(f"Merged data shape {dfm.shape}")
        return dfm

    def process_features(self, req_df: pd.DataFrame, err_df: pd.DataFrame) -> pd.DataFrame:
        """
        Полная обработка: извлечение datetime, добавление лагов, объединение датафрейма
        
        Args:
            req_df: Исходный датафрейм requests.
            err_df: Исходный датафрейм errors.
            
        Returns:
            Готовый DataFrame с признаками, индексированный по datetime.
        """

        logger.info("Starting feature engineering")
        r = self.extract_datetime_features(req_df)
        e = self.extract_datetime_features(err_df)
        r_l = self.add_lags(r, 'req')
        e_l = self.add_lags(e, 'err')
        merged = self.merge_dataframes(r_l, e_l)
        merged.set_index('datetime', inplace=True)
        logger.info(f"Features ready: {merged.shape}")
        return merged

    """
    ФУНКЦИИ для работы с одним датасетом
    """

    def process_single_dataset(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Обработка признаков одного датафрейма.
        
        Args:
            df: Исходный датафрейм (requests или errors)
            dataset_type: 'req' для requests или 'err' для errors
            
        Returns:
            Готовый DataFrame с признаками, индексированный по datetime
        """
        logger.info(f"Starting  feature engineering for single dataset {dataset_type}")

        df_with_time = self.extract_datetime_features(df)

        df_with_lags = self.add_lags(df_with_time, dataset_type)

        df_with_lags.set_index('datetime', inplace=True)

        logger.info(f"Single dataset shape: {df_with_lags.shape}")
        return df_with_lags
