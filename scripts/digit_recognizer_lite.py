from src.dataset_loader.kaggle_dataset_loader import KaggleDatasetLoader
from src.dataset.dig_rec_dataset import DigRecDataSet
import logging
from src.model import Model
from src.model.cost_function.squared_cost import SquaredCost
from src.model.layer.fully_connected_layer import FullyConnectedLayer
from src.model.layer.activation.relu import Relu
from src.model.layer.activation.soft_max import SoftMax
from src.model.layer.max_pool_2d import MaxPool2D
from src.model.layer.conv2 import Conv2D
from src.model.layer.optimizer.adam import Adam

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def main():
    loader = KaggleDatasetLoader()
    path = loader.download("digit-recognizer", "data/")
    dataset = DigRecDataSet(path, 0.1)

    input_shape = dataset.input_shape()
    output_shape = dataset.output_shape()

    model = Model(input_shape, SquaredCost)
    reg_lambda = 0.01

    model.add_layer(Conv2D(3, 3, Relu, SquaredCost, reg_lambda, Adam()))
    model.add_layer(MaxPool2D(2))

    model.add_layer(Conv2D(3, 8, Relu, SquaredCost, reg_lambda, Adam()))
    model.add_layer(MaxPool2D(2))

    model.add_layer(Conv2D(3, 24, Relu, SquaredCost, reg_lambda, Adam()))

    model.add_layer(FullyConnectedLayer([50], Relu, SquaredCost, reg_lambda, Adam()))

    model.add_layer(
        FullyConnectedLayer(output_shape, SoftMax, SquaredCost, reg_lambda, Adam())
    )

    model.summary()

    for i in range(10000):
        if i % 10 == 0:
            pred = model.run(dataset.test.X)
            accuracy = dataset.evaluate(pred)
            logging.info(f"accuracy {accuracy} at iteration {i}")

        X, Y = dataset.get_training_batch(1000)
        model.gradiend_descent(X, Y)


if __name__ == "__main__":
    main()
