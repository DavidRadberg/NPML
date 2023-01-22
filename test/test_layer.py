from src.model.layer.random_layer import RandomLayer
from src.model.layer.activation.relu import Relu


def test_shape():
    layer = RandomLayer(10, Relu())
    layer.random_init(10)
    assert layer.input_size == 10
    assert layer.output_size == 10
