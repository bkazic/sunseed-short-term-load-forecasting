from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from sklearn import preprocessing
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

class PyBrainNN:
    """ PyBrain's Back Propagation Neural Network wrapper """
    
    # TODO: could this optional parameters be passed in a better way? With **kwargs? Problem is that later two different 
    #       functions uses **kwargs and I dont know which parameter should be passed to each function.
    def __init__(self, hidden_size=2, bias=True, learningrate=0.01, momentum=0.0, maxEpochs=None, verbose=False, normalize=True, **kwargs):
        self.hidden_size = hidden_size
        self.bias = bias
        self.learningrate = learningrate
        self.momentum = momentum
        self.maxEpochs = maxEpochs
        self.verbose = verbose
        self.normalize = normalize
    
    # All scikit predictors need to have this method.
    def get_params(self, deep=True):
        return self.__dict__
        
    # Set the parameters of this estimator.
    def set_params(self, **kwargs):
        self.__init__(**kwargs)
    
    # Train model using data as training set, adn target as target values
    def fit(self, data, target, **kwargs):
        # Create PyBrain datasets and normalize them
        train_ds = self.convertToNNData(data, target, self.normalize)
                
        # Create PyBrain net
        self.net = buildNetwork(train_ds.indim, self.hidden_size, train_ds.outdim, bias=self.bias, **kwargs)
        
        # Create PyBrain trainer - Backprop in this case
        trainer = BackpropTrainer(self.net, train_ds, learningrate=self.learningrate, momentum=self.momentum, **kwargs)
        trainer.trainUntilConvergence(maxEpochs=self.maxEpochs, **kwargs)
    
    # Make prediction for input data on trained net
    def predict(self, data):
        # Create PyBrain datasets and normalize them
        data_ds = self.convertToNNData(data, np.zeros(data.shape[0]), self.normalize)     
        return self.net.activateOnDataset(data_ds)    
    
    # Returns the coefficient of determination R2 of prediction
    def score(self, data, true_values):
        # Create PyBrain datasets and normalize them
        data_ds = self.convertToNNData(data, true_values, self.normalize)
        return r2_score(true_values, self.net.activateOnDataset(data_ds))
    
    # Converts pandas dataframe or numpy array to pybrain SupervisedDataSet
    def convertToNNData(self, data, target, normalize=True):
        # Check if data is dataframe and convert to numpy array
        if isinstance(data, pd.DataFrame):
            data = data.values
        if isinstance(target, pd.DataFrame):
            target = target.values

        if (normalize):
            preprocessing.MinMaxScaler(copy=False).fit_transform(data)
            #data = preprocessing.MinMaxScaler().fit_transform(data)

        # Initialize pybrain dataset
        ds = SupervisedDataSet(data.shape[1], 1)

        # Use comprehension list instead
        [ds.addSample(x, y) for x, y in zip(data, target)]      
        return ds
