# Tensorflow Layer Normalization
=================================
Tensorflow implementation of [Layer Normalization](https://arxiv.org/abs/1607.06450).

This implementation contains:

1. Layer Normalization for GRU
    
2. Layer Normalization for LSTM
	- Currently normalizing c causes lot of nan's in the model, thus commenting it out for now.

![model_demo](./assets/model_gru1.png)




Prerequisites
-------------

- Python 2.7 or Python 3.3+
- [NLTK](http://www.nltk.org/)
- [TensorFlow](https://www.tensorflow.org/) >= 0.9

MNIST
-----
To evaluate the new model, we train it on MNIST. Here is the model and results using Layer normalized GRU

![histogram](./assets/model_gru3.png)


![scalar](./assets/model_gru4.png)


Usage
-----

To train a mnist model with Ubuntu dataset:

    $ ppython mnist.py --hidden 128 summaries_dir log/ --cell_type LNGRU
    
 cell_type = [LNGRU, LNLSTM, LSTM , GRU, BasicRNN]
    



Todo
-----
1. Add attention based models ( in progress ). 
