from torch import nn

from .base_discard_net import BaseDiscardNet


class DiscardNetV1(BaseDiscardNet):
    """
    Neural network for discarding cards.

    ------

    Network structure:

    INPUT
    -> Linear(128), ReLU, Dropout(0.3)
    -> Linear(64), ReLU, Dropout(0.3)
    OUTPUT
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(64, self.OUTPUT_SIZE)
        )
        self.net.to(self.device)


__all__ = ['DiscardNetV1']
