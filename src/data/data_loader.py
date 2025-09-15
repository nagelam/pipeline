import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Загрузка данных, проверка на пропуски, перенос индекса в колонку"""

    def __init__(self, config: dict):
        self.config = config
        self.data_config = config['data']

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Сортируем по индексу датасет, сбрасываем индекс c временем в колонку datetime
        
        Args:
            df: датафрейм для которого эта операция выполняется
            
        Returns:
            датафрейм с добавленной колонкой из времени
        """

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df = df.reset_index()
        df = df.rename(columns={'index': 'datetime'})

        return df

    def check_missing_values(self, df: pd.DataFrame, name: str) -> None:
        """
        Проверяем есть ли у нас пропущенные значения, если есть то  
         предупреждаем в какой колонке
        
        Args:
            df: датафрейм в котором проверяем пропущенные значения
            name: requests, errors тип датафрейма для печати
        """
        missing = df.isna().sum()
        if missing.sum() > 0:
            # говорим в каком датасете пропуск
            logger.warning(f"Missing values in {name}")
            for col, count in missing.items():
                if count > 0:
                    # в какой колонке и сколько
                    logger.warning(f" {col}: {count}")
        else:
            logger.info(f"No missing values in {name}")

    def load_data(self, file_key: str, old_col_key: str, new_col_key: str, log_name: str) -> pd.DataFrame:
        """
        метод для загрузки и предобработки данных для запросов и ошибок
        
        Args:
            file_key: Ключ для пути к файлу в конфиге
            old_col_key: НЕ ИСПОЛЬЗУЕТСЯ ТАК КАК В ФУНКЦИИ МЫ ПРОХОДИМСЯ ПО возможным колонкам
            new_col_key: Ключ для нового имени колонки в конфиге
            log_name: Имя для логирования (например, requests)
        Returns:
            Предобработанный DataFrame
        """

        file_path = self.data_config[file_key]

        logger.info(f"Loading {log_name} data from {file_path}")

        df = pd.read_csv(
            file_path,
            index_col=0,
            parse_dates=True,
            compression="zstd",
        )
        new_col = self.data_config[new_col_key]
        candidates = [c for c in df.columns if str(c).lower() not in ('datetime', 'index')]
        if len(candidates) == 1:
            chosen = candidates
            logger.info(f"[{log_name}] Renaming column '{chosen}' to '{new_col}'")
            df = df.rename(columns={chosen[0]: new_col})
        else:
            raise ValueError(
                f"{log_name} Expected  one column, found {len(candidates)}: {candidates}"
            )

        # Сбрасываем индекс и переводим время в колонку
        df = self.preprocess_dataframe(df)

        logger.info(f"Loaded {log_name} data: {df.shape}")
        return df

    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Загружает оба датасета из файлов, проверяет их на пропуски
        
        Returns:
            оба датасета requests_df, errors_df
        """
        requests_df = self.load_data(
            'requests_file',
            'requests_target_col',
            'renamed_requests_col',
            'requests'
        )
        errors_df = self.load_data(
            'errors_file',
            'errors_target_col',
            'renamed_errors_col',
            'errors'
        )

        # проверяем на пропуски
        self.check_missing_values(requests_df, "requests")
        self.check_missing_values(errors_df, "errors")

        return requests_df, errors_df

    """
    ФУНКЦИИ для работы с одним датасетом
    """

    def load_requests_only(self) -> pd.DataFrame:
        """
        Загружает только requests данные.
        
        Returns:
            requests DataFrame
        """
        logger.info("Loading requests data only")

        requests_df = self.load_data(
            'requests_file',
            'requests_target_col',
            'renamed_requests_col',
            'requests'
        )

        self.check_missing_values(requests_df, "requests")
        logger.info("Requests data loaded")

        return requests_df

    def load_errors_only(self) -> pd.DataFrame:
        """
        Загружает только errors данные.
        
        Returns:
            errors DataFrame
        """
        logger.info("Loading errors data only")

        errors_df = self.load_data(
            'errors_file',
            'errors_target_col',
            'renamed_errors_col',
            'errors'
        )

        self.check_missing_values(errors_df, "errors")
        logger.info("Errors data loaded")

        return errors_df
