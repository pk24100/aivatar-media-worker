from enum import Enum, auto


# Enum defining strategies for skipping layers during inference.
class SkipLayerStrategy(Enum):
    AttentionSkip = auto()
    AttentionValues = auto()
    Residual = auto()
    TransformerBlock = auto()
