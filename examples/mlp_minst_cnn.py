import numpy as np
import sklearn.datasets
import deeppy as dp
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s %(message)s',
)

def run():
	# Fetch data
	mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home='./data')

	

	X = mnist.data/255.0
    y = mnist.target

    n = y.size
    shuffle_idxs = np.random.random_integers(0, n-1, n)

    X = X[shuffle_idxs, ...]
    y = y[shuffle_idxs, ...]

	
	n_test = 150
    n_valid = 15
    n_train = 75


    print ("n_train ", n_train)
    print ("n_valid ", n_valid)
    print ("n_test ", n_test)

	X_train = np.reshape(X[:n_train], (-1, 1, 28, 28))
	y_train = y[:n_train]

	X_valid = np.reshape(X[n_train:n_train+n_valid], (-1, 1, 28, 28))
	y_valid = y[n_train:n_train+n_valid]

	X_test = np.reshape(X[n_train+n_valid:], (-1, 1, 28, 28))
	y_test = y[n_train+n_valid:n_train+n_valid+n_test]
	n_classes = np.unique(y_train).size
	
	# Setup multi-layer perceptron
	nn = dp.NeuralNetwork(
	    layers=[
	        dp.Convolutional(
	            n_output=12,
	            filter_shape=(5,5),
	            weights=dp.NormalFiller(sigma=0.1),
	            weight_decay=0.00001,
	        ),
	        dp.Activation('relu'),
	        dp.Flatten(),
	        dp.FullyConnected(
	            n_output=n_classes,
	            weights=dp.NormalFiller(sigma=0.1),
	            weight_decay=0.00001,
	        ),
	        dp.MultinomialLogReg(),
	    ],
	)
	
	# Train neural network
	trainer = dp.StochasticGradientDescent(
	    batch_size=15, learn_rate=0.05, learn_momentum=0.9, max_epochs=15
	)
	trainer.train(nn, X_train, y_train, X_valid, y_valid)

    # Evaluate on test data
    error = nn.error(X_test, y_test)
    print('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run()	