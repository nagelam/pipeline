import logging
from typing import List, Dict, Any

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tcn import TCN  # pip install keras-tcn

logger = logging.getLogger(__name__)


class TCNForecaster:
    """TCN модель для совместного прогноза двух рядов"""

    def __init__(self, config: dict):
        self.config = config
        self.training_config = config['training']

    def build_model(
            self,
            layers_config: List[Dict[str, Any]],
            timesteps: int,
            n_features: int,
            learning_rate: float,
            n_outputs: int = 2
    ) -> Sequential:
        """
        Строит TCN с двухмерным выходом

        Вид поддерживаемой архитектуры:
        поддерживает многослойные ТСН с набором которые укзанны в tcn_layer
        после каждого слоя может быть dropout_after

        Args:
            layers_config: список конфигов
            timesteps: длина окна по времени
            n_features: число входных признаков
            learning_rate: скорость обучения
            n_outputs: размерность выхода (2)
        """
        model = Sequential()
        model.add(Input(shape=(timesteps, n_features)))

        for idx, layer_conf in enumerate(layers_config):
            rs = layer_conf.get('return_sequences', idx < len(layers_config) - 1)

            tcn_layer = TCN(
                nb_filters=layer_conf.get('nb_filters', 64),
                kernel_size=layer_conf.get('kernel_size', 3),
                nb_stacks=layer_conf.get('nb_stacks', 1),
                dilations=layer_conf.get('dilations', [1, 2, 4, 8, 16]),
                padding=layer_conf.get('padding', 'causal'),
                use_skip_connections=layer_conf.get('use_skip_connections', True),
                dropout_rate=layer_conf.get('dropout', 0.0),
                return_sequences=rs,
                use_batch_norm=layer_conf.get('use_batch_norm', False),
                activation=layer_conf.get('activation', 'relu')
            )
            model.add(tcn_layer)

            if 'dropout_after' in layer_conf and layer_conf['dropout_after'] > 0:
                model.add(Dropout(layer_conf['dropout_after']))

        last = layers_config[-1]
        if 'dense_units' in last:
            model.add(Dense(last['dense_units'], activation='relu'))

        model.add(Dense(n_outputs, name='output'))  # Dense(2)
        model.compile(optimizer=Adam(learning_rate), loss='mae')
        # logger.info(model.summary())
        return model
