from config import hyperparameters
import dataset
import neuralnet
import coloredlogs, logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')
coloredlogs.install(level='DEBUG')

iterations = hyperparameters.get('iterations')
learning_rate = hyperparameters.get('learning_rate')
hidden_nodes = hyperparameters.get('hidden_nodes')
output_nodes = hyperparameters.get('output_nodes')
np.random.seed(1)

data = dataset.Dataset('data/hours.csv')
train_features, train_targets = data.return_train_data()
test_features, test_targets = data.return_test_data()
val_features, val_targets = data.return_val_features()

N_i = train_features.shape[1]
network = neuralnet.NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)
losses = {'train':[], 'validation':[]}

logger.info("Initializing Neural Network Training")
for ii in range(iterations):
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.ix[batch].values, train_features.ix[batch]['cnt']
    network.train(X, y)

    train_loss = neuralnet.MSE(network.run(train_features).T, train_targets['cnt'].values)
    val_loss = neuralnet.MSE(network.run(val_features).T, val_targets['cnt'].values)
    logger.debug("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()
plt.savefigure("loss.jpg")

fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['cnt']
predictions = network.run(test_features).T*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)
plt.savefigure("prediction.jpg")
