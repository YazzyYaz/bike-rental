import dataset
import neuralnet
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from miniflow import *
from config import hyperparameters, logger, dataset_link

class Session(object):
    """Training Neural Network Session Creator"""
    def __init__(self, mode="numpy"):
        self.mode = mode
        self.iterations = hyperparameters.get('iterations')
        self.learning_rate = hyperparameters.get('learning_rate')
        self.hidden_nodes = hyperparameters.get('hidden_nodes')
        self.output_nodes = hyperparameters.get('output_nodes')
        np.random.seed(1)

        self.data = dataset.Dataset(dataset_link)
        self.train_features, self.train_targets = self.data.return_train_data()
        self.test_features, self.test_targets = self.data.return_test_data()
        self.val_features, self.val_targets = self.data.return_val_features()
        self.scaled_features = self.data.return_scaled_features()
        self.rides = self.data.return_rides()
        self.test_data = self.data.test_data

        if self.mode == "numpy":
            self.__initiate_numpy_neural_net()
        else:
            self.__initiate_miniflow_neural_net()

    def __initiate_numpy_neural_net(self):
        N_i = self.train_features.shape[1]
        network = neuralnet.NeuralNetwork(N_i, self.hidden_nodes, self.output_nodes, self.learning_rate)
        losses = {'train':[], 'validation':[]}

        logger.info("Initializing Neural Network Training")
        for ii in range(self.iterations):
            batch = np.random.choice(self.train_features.index, size=128)
            X, y = self.train_features.ix[batch].values, self.train_targets.ix[batch]['cnt']
            network.train(X, y)

            train_loss = neuralnet.MSE(network.run(self.train_features).T, self.train_targets['cnt'].values)
            val_loss = neuralnet.MSE(network.run(self.val_features).T, self.val_targets['cnt'].values)
            if (ii % 50 == 0):
                progress = str("\rProgress: {:2.1f}".format(100 * ii/float(self.iterations)) \
                             + "% ... Training loss: " + str(train_loss)[:5] \
                             + " ... Validation loss: " + str(val_loss)[:5])
                logger.debug(progress)

            losses['train'].append(train_loss)
            losses['validation'].append(val_loss)

        logger.info("Training Complete")
        logger.info("Generating Loss Plot")
        plt.plot(losses['train'], label='Training loss')
        plt.plot(losses['validation'], label='Validation loss')
        plt.legend()
        _ = plt.ylim()
        plt.savefig("assets/loss.png")
        logger.info("Loss Plot Generated")

        logger.info("Generating Prediction Plot")

        fig, ax = plt.subplots(figsize=(8,4))

        mean, std = self.scaled_features['cnt']
        predictions = network.run(self.test_features).T*std + mean
        ax.plot(predictions[0], label='Prediction')
        ax.plot((self.test_targets['cnt']*std + mean).values, label='Data')
        ax.set_xlim(right=len(predictions))
        ax.legend()

        dates = pd.to_datetime(self.rides.ix[self.test_data.index]['dteday'])
        dates = dates.apply(lambda d: d.strftime('%b %d'))
        ax.set_xticks(np.arange(len(dates))[12::24])
        _ = ax.set_xticklabels(dates[12::24], rotation=45)
        plt.savefig("assets/prediction.png")
        logger.info("Prediction Plot Generated")

    def __initiate_miniflow_neural_net(self):
        inputs, weights, bias = Input(), Input(), Input()
        f = Linear(inputs, weights, bias)
        feed_dict = {
            inputs: [6, 14, 3],
            weights:[0.5, 0.25, 1.4],
            bias: 2
        }
        sorted_nodes = topological_sort(feed_dict)
        output = forward_pass(f, sorted_nodes)
        logger.info(output)
