import os
import numpy as np
import pickle


save_data_path="C:/Users/Torbjørn/Google Drev/UNI/MastersProject/EverythingkwantRL/saved_data"
# save_data_path="C:/Users/Torbjørn/Google Drev/UNI/MastersProject/Simulation"
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

def load_cma_output(data_path=None,run_number=None):
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
        
    if run_number==None:
        folders=list(os.walk(path+"/outcmaes/"))[0][1]
        folders_as_int=[int(f) for f in folders]
        latest_run=max(folders_as_int)
        path+='/outcmaes/{}/'.format(latest_run)
        
    else:
        path+='/outcmaes/{}/'.format(run_number)
    xs=np.loadtxt(path+"xrecentbest.dat",skiprows=1)
    return xs[:,4],xs[:,5:]

class datahandler():
    def __init__(self,experiment_name,QPC=None,data_path=None):
        
        if data_path==None:
            self.data_path=save_data_path+'/'
        else:
            self.data_path=data_path+'/'
            
        self.fname=experiment_name+".pkl"
        
        if self.fname in list(os.walk(self.data_path))[0][2]:
            self.dict=self.load_dict()
            if QPC!=None:
                for key in parameters:
                    if not self.dict[key]==QPC.__dict__[key]:
                        raise Exception("Parameters do not match at key: "+key +" ,with {} in existing dict, and {} in QPC.__dict__".format(self.dict[key],QPC.__dict__[key]))
                        
        else:
            self.dict=self.new_dict(QPC)

        
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
            key+='{:.3f}_'.format(x)
        return key[:-1]
    



    
    
    
    
    # def load_closest_data(self,voltages):
    #     "OLD - doesnt work with dictionaries yet"
    #     all_files=self.get_file_names(self.qpctype)
    #     arr=[]
    #     for fname in all_files:
    #         arr.append(fname[:-4].split('_'))
    #     arr=np.array(arr,dtype=float)
    #     idx=np.linalg.norm(arr-np.array(voltages),axis=1).argmin()
    #     return self.load_data(arr[idx,:],self.qpctype)
    
    # def check_data(self,voltages):
    #     "OBSOLETE - isn't needed with "
    #     fname=self.fname
    #     for V in voltages:
    #         fname+='{:.2f}_'.format(V)
    #     fname=fname[:-1]+'.npy'
    #     if os.path.isfile(fname):
    #         return True
    #     else:
    #         return False

    # def get_file_names(self):
    #     "OBSOLETE"
    #     return list(os.walk(self.data_path+self.qpctype+"/"))[0][2]
                   
    # def load_data(self,voltages):
    #     "OBSOLETE"
    #     fname=self.fname
    #     for V in voltages:
    #         fname+='{:.2f}_'.format(V)
    #     fname=fname[:-1]+'.npy'
    #     return np.load(fname)
    
    # def save_data(self,result,voltages):
    #     "OBSOLETE"
    #     fname=self.fname
    #     for V in voltages:
    #         fname+='{:.2f}_'.format(V)
    #     fname=fname[:-1]+'.npy' 
    #     np.save(fname,result)
        

    





    