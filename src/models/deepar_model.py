import logging
from typing import Dict, Any

from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.metrics import MultivariateNormalDistributionLoss, MultiLoss

logger = logging.getLogger(__name__)


class DeepARForecaster:
    """Строит и возвращает DeepAR модель (2 таргета)"""

    def build_model(
            self,
            layers_config: list[Dict[str, Any]],
            dataset: TimeSeriesDataSet,
            learning_rate: float
    ) -> DeepAR:

        cfg = layers_config[0]

        hidden_size = int(cfg.get("hidden_size", 32))
        rnn_layers = int(cfg.get("rnn_layers", 2))
        dropout = float(cfg.get("dropout", 0.1))

        model = DeepAR.from_dataset(
            dataset,
            hidden_size=hidden_size,
            rnn_layers=rnn_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            loss=MultiLoss([
                MultivariateNormalDistributionLoss(rank=2),
                MultivariateNormalDistributionLoss(rank=2),
            ]),
            optimizer="adam"
        )

        logger.info(
            f"DeepAR built, hidden_size={hidden_size}, "
            f"rnn_layers={rnn_layers}, dropout={dropout}"
        )

        return model
