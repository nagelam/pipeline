import numpy as np
import pandas as pd
import logging
import gc

import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class LSTMForecaster:
    """LSTM модель для прогнозирования временных рядов"""

    def __init__(self, config: dict):
        """
        Инициализация LSTM модели для прогнозирования.
        
        Args:
            config: Словарь конфигурации с настройками модели и обучения.
        """

        self.config = config
        self.training_config = config['training']

    def build_model(
            self,
            layers_config: List[Dict[str, Any]],
            timesteps: int,
            n_features: int,
            learning_rate: float
    ) -> Sequential:
        """
        Строит модель LSTM на основе конфигурации слоёв

        Вид поддерживаемой архитектуры:
        N lstm c числом units, return_sequences и без дропаут
        Дропаут
        dense_units
        
        Args:
            layers_config: Список словарей с конфигурацией слоёв LSTM
            timesteps: Гиперпараметр lstm
            n_features: Количество признаков
            learning_rate: Скорость обучения 
            
        Returns:
            Скомпилированная модель Sequential
        """

        model = Sequential()
        model.add(Input(shape=(timesteps, n_features)))
        for layer_conf in layers_config:
            model.add(
                LSTM(
                    units=layer_conf['units'],
                    return_sequences=layer_conf.get('return_sequences', False)
                )
            )
            if 'dropout' in layer_conf:
                model.add(Dropout(layer_conf['dropout']))
        last = layers_config[-1]
        if 'dense_units' in last:
            model.add(Dense(last['dense_units'], activation='relu'))
        model.add(Dense(1, name='output'))
        model.compile(optimizer=Adam(learning_rate), loss='mae')
        return model
