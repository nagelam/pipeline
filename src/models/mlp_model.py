import numpy as np
import pandas as pd
import logging
import gc
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, QuantileLoss
from pytorch_forecasting.models.mlp import DecoderMLP
from lightning.pytorch.callbacks import EarlyStopping

import warnings

warnings.filterwarnings("ignore")

logging.getLogger("pytorch_forecasting").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class DecoderMLPForecaster:
    """
     DecoderMLP использующий пайторч
    """

    def __init__(self, config: dict):
        """
        Инициализация DecoderMLPForecaster.

        Args:
            config: Словарь с конфигурацией модели и обучения.
        """
        self.config = config
        self.training_config = config['training']

    def build_model(self,
                    layers_config: list[Dict[str, Any]],
                    max_encoder_length: int,
                    dataset: TimeSeriesDataSet,
                    learning_rate: float) -> DecoderMLP:
        """
        Построение модели DecoderMLP на основе датасета и конфигурации.

        Вид поддерживаемой архитектуры:
        MLP с hidden_size, n_layers и dropout
        Тоесть не поддерживает несколько слоев

        Args:
            layers_config: Конфигурация слоёв
            max_encoder_length: Максимальная длина энкодера
            dataset: Подготовленный TimeSeriesDataSet.
            learning_rate: Скорость обучения.
        Returns:
            DecoderMLP: Построенная модель.
        """
        config = layers_config[0]

        hidden_size = int(config.get('hidden_size', 64))
        dropout = float(config.get('dropout', 0.1))

        model = DecoderMLP.from_dataset(
            dataset,
            hidden_size=hidden_size,
            dropout=dropout,
            learning_rate=learning_rate,
            loss=QuantileLoss()
        )

        logger.info(f"DecoderMLP with hidden_size={hidden_size},dropout={dropout}")

        return model
