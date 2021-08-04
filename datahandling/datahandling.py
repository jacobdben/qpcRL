import os
import numpy as np
import pickle
import json


#save_data_path="C:/Users/Torbjørn/Google Drev/UNI/MastersProject/EverythingkwantRL/saved_data"
# save_data_path="C:/Users/Torbjørn/Google Drev/UNI/MastersProject/Simulation"
# save_data_path="/nbi/user-scratch/f/fxg433/projects/saved_data2"
save_data_path="F:/qcodes_data/BBQPC3/saved_data"
# save_data_path="F:/qcodes_data/BBQPC_2021/saved_data"
# save_data_path="F:/qcodes_data/BBQPC2_2021/saved_data"
Vs=['V%i'%i for i in range(1,12)]
parameters=['phi','salt','U0','energy','t']
parameters.extend(Vs)


def save_optimization_dict(name,optimization_dict):
    fname=save_data_path+'/'+"optimization_runs/"+name+".pkl"
    num=1
    while fname in list(os.walk(save_data_path+"/"+"optimization_runs/"))[0][2]:
        fname=save_data_path+'/'+"optimization_runs/"+name+"_{}".format(num)+".pkl"
        num+=1
    file=open(fname,"wb")
    pickle.dump(optimization_dict,file)
    file.close
    
def load_optimization_dict(name):
    file=open(save_data_path+'/'+"optimization_runs/"+name+".pkl","rb")
    output=pickle.load(file)
    file.close
    return output

def load_cma_data(run_id=None,data_path=None):
    """
    Parameters
    ----------
    data_path : path to, but not including, outcmaes directory, if not provided will default to save_data_path
    run_number : The number of the run, integer. if None provided, will return latest run

    Returns
    -------
    numpy array
        best fitness in iteration, sorted by iteration
    numpy array
        best solution in iteration, sorted by iteration

    """
    
    if data_path==None:
        path=save_data_path
    else:
        path=data_path
        
    if run_id==None:
        folders=list(os.walk(path+"/outcmaes/"))[0][1]
        folders_as_int=[int(f) for f in folders]
        latest_run=max(folders_as_int)
        path+='/outcmaes/{}/'.format(latest_run)
        
    else:
        path+='/outcmaes/{}/'.format(run_id)
        


    print("data loaded from:")
    print(path)
    
    with open(path+'datadict.txt','rb') as file:
        datadict=json.load(file)
        
    
    return datadict




def unpack_cma_data(datadict,starting_point=False):
    if starting_point:
        start_loss=datadict['starting_point']['measurements']['0']['val']
        start_staircase=datadict['starting_point']['measurements']['0']['staircase']
        start_voltages=datadict['starting_point']['measurements']['0']['voltages']
        return (start_loss,start_staircase,start_voltages)
    
    loss=[]
    staircases=[]
    voltages=[]
    if 'x' in datadict['measurements']['0'].keys():
        x=True
        xs=[]
    else:
        x=False
        xs=None

    for key in datadict['measurements'].keys():

        loss.append(datadict['measurements'][key]['val'])
        staircases.append(datadict['measurements'][key]['staircase'])
        voltages.append(datadict['measurements'][key]['voltages'])
        
        if x:
            xs.append(datadict['measurements'][key]['x'])
      
    return loss,staircases,voltages,xs

def save_pkl(object_instance,run_id,filename,data_path=save_data_path):
    with open(data_path+"/outcmaes/{}/".format(run_id)+filename+".pkl",'wb') as file:
        pickle.dump(object_instance,file)
    
def load_pkl(run_id,filename,data_path=save_data_path):
    with open(data_path+"/outcmaes/{}/".format(run_id)+filename+".pkl",'rb') as file:
        output=pickle.load(file)
    return output
    
    
    
class datahandler():
    def __init__(self,experiment_name,QPC=None,data_path=None):
        
        if data_path==None:
            self.data_path=save_data_path+'/'
        else:
            self.data_path=data_path+'/'
            
        
    def save_datahandler(self,):
        file=open(self.data_path+self.fname,"wb")
        pickle.dump(self.dict,file)
        file.close

    def load_dict(self,):
        file=open(self.data_path+self.fname,"rb")
        output=pickle.load(file)
        file.close
        return output
    
    def new_dict(self,QPC):
        new_dict={key : QPC.__dict__[key] for key in parameters}
        new_dict['measurements']={}
        return new_dict
    
    def check_measurement(self,measurement):
        key=self.make_key(measurement)
        return (key in self.dict['measurements'])
    
    def save_measurement(self,measurement,data):
        key=self.make_key(measurement)
        self.dict['measurements'][key]=data
        
    def load_measurement(self,measurement):
        "Can work with either a list/numpy array of the measurement parameters or the key"
        if isinstance(measurement,list) or isinstance(measurement,np.ndarray):
            key=self.make_key(measurement)
            return self.dict['measurements'][key]
        elif isinstance(measurement,str):
            return self.dict['measurements'][measurement]
    
    def make_key(self,measurement):
        key=''
        for x in measurement:
            key+='{:.4f}_'.format(x)
        return key[:-1]
    
    def read_data(self,):
        X=[]
        Y=[]
        for key in self.dict['measurements'].keys():
            vals=key.split("_")
            vals=[float(val) for val in vals]
            X.append(vals)
            Y.append(self.dict['measurements'][key])
        return np.array(X),np.array(Y).reshape((len(Y),1))
    




    





    
