import numpy as np
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
    x = x.astype(dp.float_)
    y = y.astype(dp.int_)
    n_train = int(0.8 * n_samples)
    x_train = x[:n_train]
    y_train = y[:n_train]
    x_test = x[n_train:]
    y_test = y[n_train:]

    scaler = dp.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Setup feeds
    batch_size = 16
    train_feed = dp.SupervisedFeed(x_train, y_train, batch_size=batch_size)
    test_feed = dp.Feed(x_test)

    # Setup neural network
    weight_decay = 1e-03
    net = dp.NeuralNetwork(
        layers=[
            dp.Affine(
                n_out=32,
                weights=dp.Parameter(dp.AutoFiller(),
                                     weight_decay=weight_decay),
            ),
            dp.ReLU(),
            dp.Affine(
                n_out=64,
                weights=dp.Parameter(dp.AutoFiller(),
                                     weight_decay=weight_decay),
            ),
            dp.ReLU(),
            dp.Affine(
                n_out=n_classes,
                weights=dp.Parameter(dp.AutoFiller()),
            ),
        ],
        loss=dp.SoftmaxCrossEntropy(),
    )

    # Train neural network
    learn_rule = dp.Momentum(learn_rate=0.01/batch_size, momentum=0.9)
    trainer = dp.GradientDescent(net, train_feed, learn_rule)
    trainer.train_epochs(n_epochs=10)

    # Evaluate on test data
    error = np.mean(net.predict(test_feed) != y_test)
    print('Test error rate: %.4f' % error)
    assert error < 0.2
