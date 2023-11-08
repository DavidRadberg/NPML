# NPML
This project implements a convolutional neural network (CNN) using only numpy. \
Training and forward pass is implemented using only numpy matrix multiplications. \
The code gives insight of the details of a CNN. \
The modular structure makes it possible to design a custom network architechture \
and to run on different kinds of training data.

## Get Started
Run <b>make setup</b> to install dependencies (using a virtual environment is recommended).

Running <b>python -m scripts.digit_recognizer</b> or <b>python -m scripts.digit_recognizer_lite</b> \
will train and test a CNN on the MNIST digit recognizer dataset https://www.kaggle.com/datasets/hojjatk/mnist-dataset.

The lite version is a smaller version of the network which will converge faster, than the full version. \
The script will continously train the model and show the current accuracy.

## Experiments
You can easily edit digit_recognizer.py to experiment with different network architechtures.

```
model.add_layer(Conv2D(5, 12, Relu, SquaredCost, reg_lambda, Adam()))
model.add_layer(MaxPool2D(2))
```
The first line specifies that we add a convolutional layer to our network. \
This layer applies a kernel of 5x5x12, to its input and outputs it to the next layer, since its width is 5 and depth is 12. \
Editing the width and depth parameter might impact accuracy and training speed.

The second line applies Max Pooling with a Stride of 2. \
This downscales the width and height of the input by 2, and can be used to improve training.

You can alter the architechtures by changing the parameters of these layers, or by removing or adding more of these them.

## Development commands
<b>make test</b> runs unit tests \
<b>make format</b> runs code formatting

## TODO
This a work in progress. Future work includes:
- Batch normalization
- Training on GPU
- Model saving
- New datasets
