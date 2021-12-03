
from sklearn.model_selection import train_test_split
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
import sys
import keras
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime


class feedforward_network():
    def __init__(self,input_layer_dim=10,load_model_name=None,data_dim=9,activation_func='tanh',hidden_layers=[100]):

        # self.compiled_model=self.make_model(input_layer_dim,data_dim,activation_func,hidden_layers,hidden_layer_neurons)
        if load_model_name!=None:
            self.model=load_model(load_model_name)
        self.activation_func=activation_func

        self.model = Sequential()

        # Add input layer
        # input layer neurons = parameters + 1
        self.model.add(Dense(input_layer_dim, input_dim=data_dim, kernel_initializer='normal', activation=activation_func))

        # Add hidden layers
        # Rule of thumb: #neurons= training data samples/(factor*(input_neurons+output_neurons))
        for i in range(len(hidden_layers)):
            self.model.add(Dense(hidden_layers[i], activation=activation_func))

        # Add output layer
        self.model.add(Dense(1, kernel_initializer='normal'))

        # Compile model
        self.loss_func=MeanSquaredError()
        self.model.compile(loss=self.loss_func, optimizer='adam')

    def train(self, X_train, Y_train, plot_history=True, ax=None,
                    validation_data=None,
                    epochs=10,
                    batch_size=None, #Defaults to 32 in Keras
                    verbose=0,
                    earlystopping=True):
        
        
        # Create options dict sent to model.fit
        options={}
        if earlystopping:
            es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=200)
            options['callbacks']=[es]
            
        if validation_data!=None:
            options['validation_data']=validation_data

            
        # Print start time
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("training started at: =", current_time)
       # Start timer
        start_time=time.perf_counter() 
        history=self.model.fit(X_train,Y_train, epochs=epochs, batch_size=batch_size, verbose=verbose,**options)
        
        
        # Stop timer
        stop_time=time.perf_counter()
        print("Training time model: %.2f seconds"%(stop_time-start_time))

        if plot_history:
            self.plot_training_history(history,ax)
            return history,ax
        return history

    def plot_training_history(self,history,ax):
        if ax==None:
            fig,ax=plt.subplots()
        if 'val_loss' in history.history.keys():
            ax.plot(history.history['val_loss'],label='Validation loss')
        ax.plot(history.history['loss'],label='Loss')
        ax.set_title('Loss History with '+self.activation_func+' function')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim((0,min(history.history['loss'])+0.05))
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        return ax

    def predict(self,X_test,Y_test,plot_result=True,ax=None):
        Y_predict=self.model.predict(X_test)
        loss=self.loss_func(Y_test, Y_predict).numpy()
        if plot_result:
            plot_ax=self.plot_prediction(Y_test,Y_predict,ax,loss)
            return Y_predict,loss,plot_ax
        return Y_predict,loss


    def plot_prediction(self,Y_test,Y_predict,ax,loss):
        if ax==None:
            fig,ax=plt.subplots()
        ax.plot(Y_test,'g',label='Measured')
        # plt.plot(test,'r--',label='KerasRegressor Prediction')
        ax.plot(Y_predict,'b-.',label='Model Prediction')
        ax.plot(Y_test-Y_predict.reshape(Y_test.shape),'r',label='Diff')
        ax.set_xlabel('Input #')
        ax.set_title('Prediction with '+self.activation_func+' activation function and loss {:.3f}'.format(loss))
        ax.set_ylabel('Conductance')
        ax.grid('on')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        return ax

    
    def save(self,model_name):
        self.model.save(model_name)
    
    
    def make_model(self,input_layer_dim,data_dim,activation_func='tanh',hidden_layers=1,hidden_layer_neurons=100):
        #UNUSED, instead incoorporated into the __init__
        self.activation_func=activation_func
        model = Sequential()
        # input layer neurons= parameters + 1
        model.add(Dense(input_layer_dim, input_dim=data_dim, kernel_initializer='normal', activation=activation_func))
        # hidden layer neurons= training data samples /(factor*(input_neurons+output_neurons))
        for i in range(hidden_layers):
            model.add(Dense(hidden_layer_neurons,activation=activation_func))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model


#ALSO INCLUDED IN DATAHANDLER
def read_data(path_to_dict,dict_name):
    with open(path_to_dict+dict_name,"rb") as file:
        dictionary=pickle.load(file)
    X=[]
    Y=[]
    for key in dictionary['measurements'].keys():
        vals=key.split("_")
        vals=[float(val) for val in vals]
        X.append(vals)
        Y.append(dictionary['measurements'][key])
    return np.array(X),np.array(Y).reshape((len(Y),1))



def split_data(X,Y,seed=1,ratio=(0.6,0.3,0.1)):
    """
    wrapper around train_test_split from sklearn.model_selection

    Parameters
    ----------
    X : X_data array
    Y : Y_data array
    seed : sets the seed for shuffling of data. The default is 1.

    ratio : Ratio of splitting (train_ratio,test_ratio,validation_ratio)
            or (train_ratio,test_ratio)
            The default is (0.6,0.3,0.1).

    Returns
    -------
    X_train
    X_test
    X_val
    Y_train
    Y_test
    Y_val

    """
    if len(ratio)==2:
        return train_test_split(X,Y,test_size=ratio[1],random_state=seed,shuffle=True)
    elif len(ratio)==3:
        X_train,X_test_and_val,Y_train,Y_test_and_val=train_test_split(X,Y,test_size=ratio[1]+ratio[2])
        X_test,X_val,Y_test,Y_val=train_test_split(X_test_and_val,Y_test_and_val,test_size=ratio[2]/(ratio[1]+ratio[2]))
        return X_train,X_test,X_val,Y_train,Y_test,Y_val


# def split_data(data,seed=1,ratio=(0.6,0.3,0.1)):
#     np.random.seed(seed)
#     random_indices=np.random.choice(data.shape[0],data.shape[0],replace=False)
#     if len(ratio)==2:
#         if len(data.shape)==1:
#             return data[random_indices[:np.floor(ratio[0]*data.shape[0])]],data[random_indices[np.floor(ratio[0]*data.shape[0]):]]
#         elif len(data.shape)==2:
#             return data[random_indices[:np.floor(ratio[0]*data.shape[0])],:],data[random_indices[np.floor(ratio[0]*data.shape[0]):],:]

#     elif len(ratio)==3:
#         if len(data.shape)==1:
#             return data[random_indices[:np.floor(ratio[0]*data.shape[0])]],data[random_indices[np.floor(ratio[0]*data.shape[0]):np.floor(ratio[0]+ratio[1]*data.shape[0])]],data[random_indices[np.floor(ratio[0]+ratio[1]*data.shape[0]):]]
#         elif len(data.shape)==2:
#             return data[random_indices[:np.floor(ratio[0]*data.shape[0])],:],data[random_indices[np.floor(ratio[0]*data.shape[0]):np.floor(ratio[0]+ratio[1]*data.shape[0])],:],data[random_indices[np.floor(ratio[0]+ratio[1]*data.shape[0]):],:]
