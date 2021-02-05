from neuralnetwork.makemodel import regression_network,split_data
from datahandling.datahandling import datahandler
dat=datahandler('ML_data_random')
dat2=datahandler('ML_data')
X,Y=dat.read_data()

X_test_2,Y_test_2=dat2.read_data()

X_train,X_test,X_val,Y_train,Y_test,Y_val=split_data(X,Y)

network=regression_network()
network.train_model(X_train,Y_train,epochs=100,validation_data=(X_val,Y_val))
# network.predict(X_test,Y_test)
network.predict(X_train,Y_train)

# res ,loss, ax=network.predict(X_test_2,Y_test_2)
