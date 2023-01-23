from src.dataset_loader.kaggle_dataset_loader import KaggleDatasetLoader
from src.dataset.dig_rec_dataset import DigRecDataSet
import logging
from src.model import Model
from src.model.cost_function.squared_cost import SquaredCost
from src.model.layer.fully_connected_layer import FullyConnectedLayer
from src.model.layer.activation.relu import Relu
from src.model.layer.activation.soft_max import SoftMax

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def main():
    loader = KaggleDatasetLoader()
    path = loader.download("digit-recognizer", "data/")
    dataset = DigRecDataSet(path, 0.1)

    input_size, output_size = dataset.train.shape()
    model = Model(input_size, SquaredCost)
    reg_lambda = 0.01
    model.add_layer(FullyConnectedLayer(20, Relu, SquaredCost, 0.01))
    model.add_layer(FullyConnectedLayer(output_size, SoftMax, SquaredCost, reg_lambda))

    for i in range(10000):
        model.gradiend_descent(dataset.train.X, dataset.train.Y, 0.1)

        if i % 10 == 0:
            pred = model.predict(dataset.test.X)
            accuracy = dataset.evaluate(pred)
            print(f"accuracy {accuracy} at epoch {i}")


if __name__ == "__main__":
    main()
