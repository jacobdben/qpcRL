from neuralnetwork.makemodel import feedforward_network,split_data
from datahandling.datahandling import datahandler
import numpy as np
import matplotlib.pyplot as plt

#randomly generated points within constraints, disorder seed=2
dat=datahandler('ML_data_random')
X,Y=dat.read_data()
X_train,X_test,X_val,Y_train,Y_test,Y_val=split_data(X,Y)

#random generated points within constraints, disorder seed=3
dat2=datahandler('ML_data_disorder_seed_3')
X2,Y2=dat2.read_data()
X_train2,X_test2,X_val2,Y_train2,Y_test2,Y_val2=split_data(X2,Y2)



# losses=[]
# epochs=[]
# hidden_layer_nodes_list=np.arange(40,80,20)
# for hidden_layer_nodes in hidden_layer_nodes_list:
hidden_layer_nodes=60
network=feedforward_network(input_layer_dim=10,hidden_layers=[hidden_layer_nodes]*2)
history,ax=network.train(X_train,Y_train,epochs=1000,plot_history=True,validation_data=(X_val,Y_val),earlystopping=True)

# history,ax=network.train(X_train2,Y_train2,epochs=100,plot_history=True,validation_data=(X_val2,Y_val2),earlystopping=True)
# network.predict(X_test,Y_test)
# network.predict(X_train,Y_train)

# model_save_name=dat.data_path+'saved_models/'+'1'
# network.save(model_save_name)
# fig,ax1=plt.subplots()
res1 ,loss, ax=network.predict(X_test2[:100],Y_test2[:100])
ax.set_title('Prediction on different disorder, without training')
ax.legend()
plt.savefig("C:/Users/Torbjørn/Google Drev/UNI/MastersProject/EverythingkwantRL/figures/"+'disorder_test3')


# furher training and then predict again to see improvement
history,ax=network.train(X_train2[:500],Y_train2[:500],epochs=100,plot_history=True,validation_data=(X_val2,Y_val2),earlystopping=True)
res2 ,loss, ax=network.predict(X_test2[:100],Y_test2[:100])
ax.set_title('Prediction on different disorder, with extra training')
ax.legend()
plt.savefig("C:/Users/Torbjørn/Google Drev/UNI/MastersProject/EverythingkwantRL/figures/"+'disorder_test4')

plt.figure()
plt.plot(Y_test2[:100],label="Measured")
plt.plot(res1,'-.',label="without extra training")
plt.plot(res2,'-*',label="with extra training")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.xlabel("Input #")
plt.ylabel("Conductance")
# losses.append(loss)
# epochs.append(len(history.history['loss']))
    
# plt.figure()
# # plt.plot(hidden_layer_nodes_list,losses)
# plt.xlabel("hidden layer nodes")
# plt.ylabel("loss")
# plt.title("loss after 500 epochs, various # hidden nodes")
# plt.grid()

# load_net=feedforward_network(load_model_name=model_save_name)
