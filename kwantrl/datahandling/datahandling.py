#%%
import os
import numpy as np
import pickle
import json


# save_data_path="C:/Users/Torbjørn/Google Drev/UNI/MastersProject/EverythingkwantRL/saved_data" 
save_data_path='/Users/qdev_26/Documents/PhD/kwantrl/kwantrl_data'
# save_data_path="/nbi/user-scratch/f/fxg433/projects/saved_data2"
# save_data_path="F:/qcodes_data/BBQPC3/saved_data"
# save_data_path="F:/qcodes_data/BBQPC_2021/saved_data"
# save_data_path="F:/qcodes_data/BBQPC2_2021/saved_data"

#%%
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

    
    
class datahandler():
    def __init__(self,data_path=None):
        
        if data_path==None:
            self.data_path=save_data_path+'/'
        else:
            self.data_path=data_path+'/'
            
    def next_outcmaes_folder_name(self,):
        if not os.path.exists(self.data_path+"outcmaes"):
            os.mkdir(self.data_path+"outcmaes")
        folders=folders=list(os.walk(self.data_path+"outcmaes/"))[0][1]
        lis=[int(f) for f in folders]
        lis.append(0)
        newfolder=self.data_path+'outcmaes/'+'{}/'.format(max(lis)+1)
        return newfolder, max(lis)+1

    def latest_outcmaes_folder_name(self,):
        if not os.path.exists(self.data_path+"outcmaes"):
            os.mkdir(self.data_path+"outcmaes")
        folders=folders=list(os.walk(self.data_path+"outcmaes/"))[0][1]
        lis=[int(f) for f in folders]
        lis.append(0)
        oldfolder=self.data_path+'outcmaes/'+'{}/'.format(max(lis))
        return oldfolder

    def save_data(self,datadict,folder=None):
        if folder==None:
            folder=self.data_path
        with open(folder+"datadict.txt",mode='w') as file_object:
            file_object.write(json.dumps(datadict))

    def load_data(self,run_id=None,folder=None):
        if run_id!=None:
            folder=f"{self.data_path}outcmaes/{run_id}/"
        elif run_id==None & folder == None:
            folder=self.latest_outcmaes_folder_name()

        with open(folder+'datadict.txt','rb') as file:
            datadict=json.load(file)
        return datadict
        
    def transform_data(self,datadict):
        return_dict={}
        #initialized dict with the correct keys
        for key in datadict['measurements']['0']:
            return_dict[key]=[]

        #fills dict lists with values
        for key in datadict['measurements']:
            for key2,value in datadict['measurements'][key].items():
                return_dict[key2].append(value)
        return return_dict

    def load_transformed_data(self,run_id=None,folder=None):
        return self.transform_data(self.load_data(run_id,folder))



    
if __name__=='__main__':
    dat=datahandler()
    folder_name,run_id=dat.next_outcmaes_folder_name()
    print(folder_name)
    print(run_id)
# %%
