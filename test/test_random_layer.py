from src.model.layer.random_layer import RandomLayer
from src.model.layer.activation.relu import Relu


def test_shape():
    layer = RandomLayer(10, Relu())
