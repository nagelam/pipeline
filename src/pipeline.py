import traceback

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging
import torch
from pathlib import Path
import json
from typing import List, Tuple

from .data.data_loader import DataLoader
from .features.feature_engineering import FeatureEngineer
from .rolling_forecast.rolling_forecast_lstm import RollingForecaster as LSTMRollingForecaster
from .rolling_forecast.rolling_forecast_tcn import RollingForecasterTCN
from .rolling_forecast.rolling_forecast_deepar import RollingForecasterDeepAR
from .rolling_forecast.rolling_forecast_mlp import RollingForecasterMLP

logger = logging.getLogger(__name__)


class TimeSeriesPipeline:
    def __init__(self, config: dict, use_decoder_mlp: bool = False, use_deepar: bool = False, use_tcn: bool = False):
        """
        Инициализация пайплайна.
        - Для 1 ряда LSTM (по умолчанию) или DecoderMLP (use_decoder_mlp=True)
        - Для 2 рядов TCN (use_tcn=True) или DeepAR (use_deepar=True).
        """
        self.config = config
        self.data_loader = DataLoader(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.use_decoder_mlp = use_decoder_mlp
        self.use_deepar = use_deepar
        self.use_tcn = use_tcn
        # Выбор модели
        if use_deepar or use_tcn:
            self.mode = "multi"
            if use_deepar:
                self.rolling_forecaster = RollingForecasterDeepAR(self.config)
                logger.info("DeepAR for 2 sequences")
            else:
                self.rolling_forecaster = RollingForecasterTCN(self.config)
                logger.info("TCN for 2 sequences")
        else:
            self.mode = "single"
            if use_decoder_mlp:
                self.rolling_forecaster = RollingForecasterMLP(self.config)
                logger.info("DecoderMLP for 1 sequence")
            else:
                self.rolling_forecaster = LSTMRollingForecaster(self.config)
                logger.info("LSTM for 1 sequence")

        self.output_dir = Path(self.config['output']['results_dir'])
        self.output_dir.mkdir(exist_ok=True)
        logger.info("Pipeline init")

    def get_target_architectures(self, target_col: str) -> Dict[str, Dict]:
        """
        Возвращает словарь доступных архитектур для указанной целевой колонки.

          Если имя target_col начинается с 'req' то ищет в config['model']['requests']
          иначе в config['model']['errors']
        Args:
            target_col: Имя целевой колонки после feature_engineer

        Returns:
            Словарь вида {arch_name: {'layers': [...]}, ...}
            То есть поддерживает случай когда например в requests несколько моделей и вернет их все
        """
        target_key = 'requests' if target_col.startswith('req') else 'errors'
        if 'model' in self.config:
            # Проверяем ключ target_key
            if target_key in self.config['model']:
                architectures = self.config['model'][target_key]
                logger.info(f"Found architectures for {target_key}: {list(architectures.keys())}")
                return architectures
        logger.warning(f"No architectures found for {target_key} Return empty dict")
        return {}

    """
    Методы для 1 ряда: LSTM и DecoderMLP
    """

    def run_forecasting_single(self, features_df: pd.DataFrame, target_col: str, architecture_name: str) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Запускает rolling forecast для 1 ряда для заданного таргета и архитектуры.

        Args:
            features_df: Датафрейм с признаками и таргетами
            target_col: Имя целевой колонки
            architecture_name: Имя архитектуры

        Returns:
            (agg_df, daily_df), agg_df содержит агрегированные метрики по окнам(число) (MAE_mean,MAE_median MSE_mean,MSE_median)
            daily_df содержит метрики по дням (MAE, MSE)
        """
        if self.mode != "single":
            raise ValueError("This method is for single-output mode only")
        model_type = "MLP" if self.use_decoder_mlp else "LSTM"
        logger.info(f"Running {model_type} for {target_col} with {architecture_name}")

        archs = self.get_target_architectures(target_col)
        if architecture_name not in archs:
            raise ValueError(f"Architecture '{architecture_name}' not found")
        layers_config = archs[architecture_name]['layers']

        requests_target = self.config['data']['renamed_requests_col']
        errors_target = self.config['data']['renamed_errors_col']
        second_target = errors_target if target_col == requests_target else requests_target
        return self.rolling_forecaster.rolling_forecast(features_df, target_col, second_target, layers_config)

    def compare_architectures_single(self, requests_archs: List[str] = None,
                                     errors_archs: List[str] = None) -> pd.DataFrame:
        """
        Сравнивает несколько архитектур для single-output по двум наборам таргетов: requests и errors
        Args:
            requests_archs: Список имен архитектур для requests если None — берутся все из конфигурацции
            errors_archs: Список имен архитектур для errors если None — берутся все из конфигурации
            В моем коде эта функция вызывается без аргументов (берется все из конфигурации)

        Returns:
            DataFrame с агрегированными метриками по архитектурам и типу таргета
        """
        if self.mode != "single":
            raise ValueError("This method is for single-output mode only")
        logger.info(f"Comparing singe output architectures for {requests_archs} and {errors_archs}")

        model_type = "MLP" if self.use_decoder_mlp else "LSTM"
        requests_target = self.config['data']['renamed_requests_col']
        errors_target = self.config['data']['renamed_errors_col']

        if requests_archs is None:
            requests_archs = list(self.get_target_architectures(requests_target).keys())
        if errors_archs is None:
            errors_archs = list(self.get_target_architectures(errors_target).keys())

        requests_df, errors_df = self.data_loader.load_all_data()
        features_df = self.feature_engineer.process_features(requests_df, errors_df)

        results = []
        for target, arch_list in [(requests_target, requests_archs), (errors_target, errors_archs)]:
            for arch in arch_list:
                print(f"Comparing {arch} {type(arch)}")
                try:
                    agg, daily = self.run_forecasting_single(features_df, target, arch)
                    result = {
                        'model_type': model_type,
                        'target_type': target.split('_')[0],
                        'architecture': arch,
                        'MAE_mean': agg['MAE_mean'].iloc[0],
                        'MSE_mean': agg['MSE_mean'].iloc[0]
                    }
                    results.append(result)

                    if self.config['output']['save_results'] and self.config['output']['save_daily_metrics']:
                        daily = daily.rename(columns={'MAE': 'MAE_mean', 'MSE': 'MSE_mean'})
                        daily.to_csv(self.output_dir / f"{target}_{arch}_{model_type.lower()}_daily.csv", index=False)
                except Exception as e:
                    logger.error(f"Error: {e}")
                    traceback.print_exc()

        comparison_df = pd.DataFrame(results)
        if self.config['output']['save_results']:
            comparison_df.to_csv(self.output_dir / f"comparison_{model_type.lower()}_single.csv", index=False)
        return comparison_df

    """
    Функция для работы с одним датасетом без мерджа
    """


    def compare_single_dataset(self, dataset_type: str) -> pd.DataFrame:
        """Тоже самое что compare_architectures_single но только для 1 ряда(запросов или ошибок)
        в dataset_type должно быть 'requests' или 'errors'
        """
        if self.mode != "single":
            raise ValueError("This method is for single-output mode only")
        model_type = "MLP" if self.use_decoder_mlp else "LSTM"

        if dataset_type == 'requests':
            target_col = self.config['data']['renamed_requests_col']
            df = self.data_loader.load_requests_only()
            feat_type = 'req'
        elif dataset_type == 'errors':
            target_col = self.config['data']['renamed_errors_col']
            df = self.data_loader.load_errors_only()
            feat_type = 'err'
        else:
            raise ValueError("Invalid dataset type")

        features_df = self.feature_engineer.process_single_dataset(df, feat_type)
        archs = self.get_target_architectures(target_col).keys()
        arch_cfgs = self.get_target_architectures(target_col)
        results = []
        for arch in archs:
            try:
                layers = arch_cfgs[arch]['layers']
                logger.info(layers)
                agg, daily = self.rolling_forecaster.rolling_forecast_single(features_df, target_col,layers)
                result = {
                    'model_type': model_type,
                    'target_type': dataset_type,
                    'architecture': arch,
                    'MAE_mean': agg['MAE_mean'].iloc[0],
                    'MSE_mean': agg['MSE_mean'].iloc[0]
                }
                results.append(result)
                daily = daily.rename(columns={'MAE': 'MAE_mean', 'MSE': 'MSE_mean'})

                if self.config['output']['save_results'] and self.config['output']['save_daily_metrics']:
                    daily.to_csv(self.output_dir / f"{dataset_type}_{arch}_{model_type.lower()}_single_daily.csv",
                                 index=False)
            except Exception as e:
                logger.error(f"Error: {e}")
                traceback.print_exc()

        comparison_df = pd.DataFrame(results)
        if self.config['output']['save_results']:
            comparison_df.to_csv(self.output_dir / f"comparison_{dataset_type}_{model_type.lower()}_single.csv",
                                 index=False)
        return comparison_df

    """
    Методы для 2 рядов: TCN и DeepAR
    """

    def run_forecasting_multi(self, features_df: pd.DataFrame, architecture_name: str) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Запускает rolling forecast для 2 рядов

        Args:
            features_df: Датафрейм с признаками и двумя таргетами
            architecture_name: Имя архитектуры для requests

        Returns:
            Кортеж (agg_df, daily_df) с агрегированными и дневными метриками:
              - agg_df: MAE_req_mean, MAE_err_mean, MSE_req_mean, MSE_err_mean
              - daily_df: по дням MAE_req, MSE_req, MAE_err, MSE_err

        Raises:
            ValueError: Если метод вызван не в режиме multi или архитектура не найдена.
        """
        if self.mode != "multi":
            raise ValueError("This method is for multi-output mode only")
        model_type = "DeepAR" if self.use_deepar else "TCN"
        logger.info(f"Running {model_type} multi-output with {architecture_name}")

        requests_target = self.config['data']['renamed_requests_col']
        archs = self.get_target_architectures(requests_target)
        if architecture_name not in archs:
            raise ValueError(f"Architecture '{architecture_name}' not found")
        layers_config = archs[architecture_name]['layers']
        errors_target = self.config['data']['renamed_errors_col']
        try:
            agg, daily = self.rolling_forecaster.rolling_forecast(features_df, requests_target, errors_target,
                                                                  layers_config)
        except Exception as e:
            traceback.print_exc()
        return (agg, daily)

    def compare_architectures_multi(self, architectures: List[str] = None) -> pd.DataFrame:
        """
        Сравнивает несколько архитектур в режиме для 2 рядов (TCN или DeepAR)
        Модели берутся из request поля конфига
        Args:
            architectures: Список имен архитектур для requests если None — берутся все из конфигурации
            В моем коде данная функция вызывается без аргументов

        Returns:
            DataFrame с агрегированными метриками по архитектурам
        """
        if self.mode != "multi":
            raise ValueError("This method is for multi-output mode only")
        logger.info(f"Running multi-output with architectures {architectures}")
        model_type = "DeepAR" if self.use_deepar else "TCN"
        requests_target = self.config['data']['renamed_requests_col']

        if architectures is None:
            architectures = list(self.get_target_architectures(requests_target).keys())

        requests_df, errors_df = self.data_loader.load_all_data()
        features_df = self.feature_engineer.process_features(requests_df, errors_df)
        results = []
        for arch in architectures:
            try:
                agg, daily = self.run_forecasting_multi(features_df, arch)
                # agg хранит еще и медианные значения
                result = {
                    'model_type': model_type,
                    'target_type': 'joint',
                    'architecture': arch,
                    'MAE_req_mean': agg['MAE_req_mean'].iloc[0],
                    'MAE_err_mean': agg['MAE_err_mean'].iloc[0],
                    'MSE_req_mean': agg['MSE_req_mean'].iloc[0],
                    'MSE_err_mean': agg['MSE_err_mean'].iloc[0]
                }
                results.append(result)

                if self.config['output']['save_results'] and self.config['output']['save_daily_metrics']:
                    req_daily = daily[['day', 'MAE_req', 'MSE_req']].rename(
                        columns={'MAE_req': 'MAE_mean', 'MSE_req': 'MSE_mean'})
                    err_daily = daily[['day', 'MAE_err', 'MSE_err']].rename(
                        columns={'MAE_err': 'MAE_mean', 'MSE_err': 'MSE_mean'})
                    req_daily['day'] = pd.to_datetime(req_daily['day']).dt.date
                    err_daily['day'] = pd.to_datetime(err_daily['day']).dt.date
                    req_daily.to_csv(self.output_dir / f"req_{arch}_{model_type.lower()}_daily.csv", index=False)
                    err_daily.to_csv(self.output_dir / f"err_{arch}_{model_type.lower()}_daily.csv", index=False)
            except Exception as e:
                logger.error(f"Error: {e}")
                traceback.format_exc()

        comparison_df = pd.DataFrame(results)
        if self.config['output']['save_results']:
            comparison_df.to_csv(self.output_dir / f"comparison_{model_type.lower()}_multi.csv", index=False)
        return comparison_df
