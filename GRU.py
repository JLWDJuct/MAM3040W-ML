
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch import optim
# from pytorchtools import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# import data
data = pd.read_csv('DATA/used/sunspots.txt', header=0)

# process data , correct types, normalised afterwards
data = data.set_index(['Month'])
data.index = pd.to_datetime(data.index)

if not data.index.is_monotonic:
    data = data.sort_index()


dataValues = data["Sunspots"]
dataValues = pd.to_numeric(dataValues)

dataMax = dataValues.max()
dataMin = dataValues.min()

dataNormalized = (dataValues - dataMin) / (dataMax - dataMin)


# create lagged TS for features

def lagged_time_series(TS, nlag):
    dataCopy = TS.copy()

    for n in range(1, nlag + 1):
        dataCopy[f"lag{n}"] = dataCopy["Sunspots"].shift(n)

    dataCopy = dataCopy.iloc[nlag:]

    return dataCopy

lag = 6

dataLagged = lagged_time_series(data, lag)

# dataLagged
# dataLagged.shape

# separate features and targets
Y = dataLagged[["Sunspots"]]

X = dataLagged.drop(columns=["Sunspots"])


# into needed sets
# TRAIN AND REMAIN SPLIT
Split1 = int(len(Y) * 0.8)

Xtrain, Xremain = X[:Split1], X[Split1:]
Ytrain, Yremain = Y[:Split1], Y[Split1:]

# TEST AND VAL SPLIT
Split2 = int(len(Yremain) * 0.5)

Xtest, Xval = Xremain[:Split2], Xremain[Split2:]
Ytest, Yval = Yremain[:Split2], Yremain[Split2:]


batchSize = 32

# Firstly it is required to convert the panda dataframes into Torch Tensor data structures so pd->np->tensor

TrainFeatures = torch.Tensor(np.array(Xtrain))
TrainTargets = torch.Tensor(np.array(Ytrain))

TrainSet = TensorDataset(TrainFeatures, TrainTargets)


ValFeatures = torch.Tensor(np.array(Xval))
ValTargets = torch.Tensor(np.array(Yval))

ValSet = TensorDataset(ValFeatures, ValTargets)


TestFeatures = torch.Tensor(np.array(Xtest))
TestTargets = torch.Tensor(np.array(Ytest))

TestSet = TensorDataset(TestFeatures, TestTargets)


# Dataloaders
TrainLoader = DataLoader(TrainSet, batch_size=batchSize, shuffle=False, drop_last=True)

ValLoader = DataLoader(ValSet, batch_size=batchSize, shuffle=False, drop_last=True)

TestLoader = DataLoader(TestSet, batch_size=batchSize, shuffle=False, drop_last=True)

Loader_one = DataLoader(TestSet, batch_size=1, shuffle=False, drop_last=True)


# First checking if GPU is available
GPUpossible = torch.cuda.is_available()

if(GPUpossible):
    print('GPU available for training')
else:
    print('GPU not available for training')


# Create RNN
class GRURNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, layer_dim, dropoutProb):
        super(GRURNN, self).__init__()

        self.hidden_dim = hidden_dim

        self.layer_dim = layer_dim

        #create layers, these are GRU not normal RNN layers

        self.gru = nn.GRU(input_size, hidden_dim, layer_dim, batch_first=True, dropout=dropoutProb)

        self.tanh = nn.Tanh()

        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):

        # Use 0 - initialzation for the hidden state:
        tempBatchSize = x.size(0)

        device = "cuda" if GPUpossible else "cpu"

        h0 = torch.zeros(self.layer_dim, tempBatchSize, self.hidden_dim).requires_grad_()
        h0 = h0.to(device)

        g_out, h0 = self.gru(x, h0.detach())

        g_out = self.tanh(g_out)

        # but need to ensure output dimensions are valid
        g_out = g_out[:, -1, :]

        output = self.fc(g_out)

        return output


class Optimization:
    # instantiate self parameters- constructor
    def __init__(self, model, loss_fn, optimizer, gradClip = 3):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.TrainLosses = []
        self.ValLosses = []

    # Create Train method
    def trainIter(self, X, Y, gradClip = 3):
        # Determine whether GPU acceleration features available
        if (GPUpossible):
            self.model.cuda()

        # Set model to train mode
        self.model.train()

        # Generate output and hidden state (i.e forward pass)
        pred = self.model(X)

        # Calculate Loss
        loss = self.loss_fn(Y, pred)

        # Perform BackPropagation
        loss.backward()

        # Update parameters via optimizer , note gradient clipping first
        nn.utils.clip_grad_norm_(self.model.parameters(), gradClip)
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()


    def train(self , TrainLoader, ValLoader, batchSize=32, numEpochs=100, featuresDim = 1):


            # cast to GPU is possible
            device = "cuda" if GPUpossible else "cpu"

            for epoch in range(1, numEpochs + 1):

                batchLosses = []

                for x_batch, y_batch in TrainLoader:

                    x_batch = x_batch.view([batchSize, -1, featuresDim]).to(device)

                    y_batch = y_batch.to(device)

                    loss = self.trainIter(x_batch, y_batch)

                    batchLosses.append(loss)

                trainingLoss = np.mean(batchLosses)

                self.TrainLosses.append(trainingLoss)

                with torch.no_grad():

                    batchValLosses = []

                    for x_val, y_val in ValLoader:

                        x_val = x_val.view([batchSize, -1, featuresDim]).to(device)

                        y_val = y_val.to(device)

                        self.model.eval()

                        pred = self.model(x_val)

                        val_loss = self.loss_fn(y_val, pred).item()

                        batchValLosses.append(val_loss)

                    validationLoss = np.mean(batchValLosses)

                    self.ValLosses.append(validationLoss)

                if (epoch <= 10) | (epoch % 50 == 0):
                    print(
                        f"[{epoch}/{numEpochs}] Training loss: {trainingLoss:.4f}\t Validation loss: {validationLoss:.4f}"
                    )

            # torch.save(self.model.state_dict())


    def evaluate(self, TestLoader, batchSize=1, featuresDim=1):

        device = "cuda" if GPUpossible else "cpu"
        # device = "cpu"

        with torch.no_grad():

            predictions = []
            values = []

            for x_test, y_test in TestLoader:

                x_test = x_test.view([batchSize, -1, featuresDim]).to(device)

                y_test = y_test.to(device)

                self.model.eval()

                pred = self.model(x_test)

                predictions.append(pred.to(device).detach().cpu().numpy())

                values.append(y_test.to(device).detach().cpu().numpy())

        return predictions, values


    def plot_losses(self):
        plt.plot(self.TrainLosses, label="Training Error")
        plt.plot(self.ValLosses, label="Validation Error")
        plt.legend()
        plt.title("A line graph showing the error over epochs")
        plt.show()
        plt.close()


# Can Finally train,test and make model
inputDim = len(Xtrain.columns)
outputDim = 1
hiddenDim = 64
nLayers = 3
batchsize = batchSize
dropout = 0.4
nEpochs = 2000
LR = 1e-3
WD = 1e-4



GRU = GRURNN(input_size = inputDim, output_size = outputDim, hidden_dim = hiddenDim, layer_dim = nLayers, dropoutProb = dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(GRU.parameters(), lr=LR, weight_decay=WD)

opt = Optimization(model=GRU, loss_fn = criterion, optimizer = optimizer)
opt.train(TrainLoader, ValLoader, batchSize = batchsize, numEpochs=nEpochs, featuresDim = inputDim)
opt.plot_losses()

predictions, values = opt.evaluate(Loader_one, batchSize = 1, featuresDim = inputDim)


# implement early stopping and retrain the model

GRU = None

nEpochs = 300

GRU = GRURNN(input_size = inputDim, output_size = outputDim, hidden_dim = hiddenDim, layer_dim = nLayers, dropoutProb = dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(GRU.parameters(), lr=LR, weight_decay=WD)

opt = Optimization(model=GRU, loss_fn = criterion, optimizer = optimizer)
opt.train(TrainLoader, ValLoader, batchSize = batchsize, numEpochs=nEpochs, featuresDim = inputDim)
opt.plot_losses()


# perfrom some prediction with model
predictions, values = opt.evaluate(Loader_one, batchSize = 1, featuresDim = inputDim)

Targets = np.concatenate(values, axis=0).ravel()
Preds = np.concatenate(predictions, axis=0).ravel()
Results = pd.DataFrame(data={"Value": Targets, "Prediction": Preds}, index=Xtest.head(len(Targets)).index)
Results = Results.sort_index()

plt.figure(figsize=(8, 5))

plt.plot(Results["Prediction"], 'r.')
plt.plot(Results["Value"], 'b.')

plt.title("Predicted results vs Actual results")
plt.xlabel('Year', fontsize=8)

plt.ylabel('Monthly average number of sunspots', fontsize=8)

plt.show()

TestError = ((Targets - Preds)**2).mean(axis=None)

print(TestError)

torch.save(GRU.state_dict(), "PureRNNmodelparam.pth")
torch.save(GRU, "GRURNNmodel.pt")