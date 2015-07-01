import deeppy as dp
from sklearn.datasets import make_classification


def test_classification():
    # Make dataset
    n_classes = 2
    n_samples = 1000
    n_features = 48
    x, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_classes=n_classes,
        n_informative=n_classes*2, random_state=1
    )

    n_train = int(0.8 * n_samples)
    n_val = int(0.5 * (n_samples - n_train))

    x_train = x[:n_train]
    y_train = y[:n_train]
    x_val = x[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    x_test = x[n_train+n_val:]
    y_test = y[n_train+n_val:]

    scaler = dp.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # Setup input
    batch_size = 16
    train_input = dp.SupervisedInput(x_train, y_train, batch_size=batch_size)
    val_input = dp.SupervisedInput(x_val, y_val)
    test_input = dp.SupervisedInput(x_test, y_test)

    # Setup neural network
    weight_decay = 1e-03
    net = dp.NeuralNetwork(
        layers=[
            dp.FullyConnected(
                n_out=32,
                weights=dp.Parameter(dp.AutoFiller(),
                                     weight_decay=weight_decay),
            ),
            dp.Activation('relu'),
            dp.FullyConnected(
                n_out=64,
                weights=dp.Parameter(dp.AutoFiller(),
                                     weight_decay=weight_decay),
            ),
            dp.Activation('relu'),
            dp.FullyConnected(
                n_out=n_classes,
                weights=dp.Parameter(dp.AutoFiller()),
            ),
        ],
        loss=dp.MultinomialLogReg(),
    )

    # Train neural network
    def val_error():
        return net.error(val_input)
    trainer = dp.StochasticGradientDescent(
        min_epochs=10, learn_rule=dp.Momentum(learn_rate=0.01, momentum=0.9),
    )
    trainer.train(net, train_input, val_error)

    # Evaluate on test data
    error = net.error(test_input)
    print('Test error rate: %.4f' % error)
    assert error < 0.2
