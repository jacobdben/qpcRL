from neuralnetwork.makemodel import regression_network,split_data
from datahandling.datahandling import datahandler
dat=datahandler('ML_data_random')
dat2=datahandler('ML_data')
X,Y=dat.read_data()

X_test_2,Y_test_2=dat2.read_data()

X_train,X_test,X_val,Y_train,Y_test,Y_val=split_data(X,Y)

network=regression_network(input_layer_dim=10)
network.train_model(X_train,Y_train,epochs=5000,validation_data=(X_val,Y_val))
# network.predict(X_test,Y_test)
# network.predict(X_train,Y_train)

model_save_name=dat.data_path+'saved_models/'+'1'
network.save_model(model_save_name)
res ,loss, ax=network.predict(X_test_2[1200:,:],Y_test_2[1200:])

# load_net=regression_network(load_model_name=model_save_name)
