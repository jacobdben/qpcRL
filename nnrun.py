from neuralnetwork.makemodel import feedforward_network,split_data
from datahandling.datahandling import datahandler
import numpy as np
import matplotlib.pyplot as plt

dat=datahandler('ML_data_random')
dat2=datahandler('ML_data')
X,Y=dat.read_data()

X_test_2,Y_test_2=dat2.read_data()

X_train,X_test,X_val,Y_train,Y_test,Y_val=split_data(X,Y)

losses=[]
epochs=[]
hidden_layer_nodes_list=np.arange(40,80,20)
for hidden_layer_nodes in hidden_layer_nodes_list:
    
    network=feedforward_network(input_layer_dim=10,hidden_layers=[hidden_layer_nodes]*2)
    history=network.train(X_train,Y_train,epochs=1000,plot_history=True,validation_data=(X_val,Y_val),earlystopping=True)
    # network.predict(X_test,Y_test)
    # network.predict(X_train,Y_train)
    
    # model_save_name=dat.data_path+'saved_models/'+'1'
    # network.save(model_save_name)
    res ,loss, ax=network.predict(X_test_2[1200:,:],Y_test_2[1200:])
    losses.append(loss)
    # epochs.append(len(history.history['loss']))
    
plt.figure()
plt.plot(hidden_layer_nodes_list,losses)
plt.xlabel("hidden layer nodes")
plt.ylabel("loss")
plt.title("loss after 500 epochs, various # hidden nodes")
plt.grid()

# load_net=feedforward_network(load_model_name=model_save_name)
