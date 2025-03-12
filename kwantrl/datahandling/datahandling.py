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
# save_data_path="/home/projects/ku_00067/scratch/kwantrl_data"

#%%
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
        
    def transform_data_legacy(self,datadict):
        return_dict={}
        #initialized dict with the correct keys
        for key in datadict['measurements']['0']:
            return_dict[key]=[]

        #fills dict lists with values
        for key in datadict['measurements']:
            for key2,value in datadict['measurements'][key].items():
                return_dict[key2].append(value)
        return return_dict

    def transform_data(self,datadict):
        return_dict={}

        num_iterations=len(datadict['measurements'].keys())
        measurements_per_iteration=len(datadict['measurements']['1'].keys())
        print(f'Dictionary contains {num_iterations} iterations with {measurements_per_iteration} measurements per iteration' )
        #initialized dict with the correct keys
        for key in datadict['measurements']['1']['0']:
            return_dict[str(key)]=[]

        #fills dict lists with values
        for iteration_dict in datadict['measurements'].values(): #iterations
            for measurement_dict in iteration_dict.values():
                for key,value in measurement_dict.items():
                    return_dict[key].append(value)
        return return_dict

    def load_transformed_data(self,run_id=None,folder=None,legacy=False):
        full_dataset=self.load_data(run_id,folder)
        if legacy:
            return self.transform_data_legacy(full_dataset) , full_dataset['starting_point']['measurements']['0']
        
        return self.transform_data(full_dataset) , full_dataset['starting_point']

    def save_qpc(self,QPC,run_id=None,folder=None):
        if run_id!=None:
            folder=f"{self.data_path}outcmaes/{run_id}/"
        elif run_id==None & folder == None:
            folder=self.latest_outcmaes_folder_name()
        
        with open(folder+'qpc.pkl','wb') as file:
            pickle.dump(QPC,file)

    def load_qpc(self,run_id=None,folder=None):
        if run_id!=None:
            folder=f"{self.data_path}outcmaes/{run_id}/"
        elif run_id==None & folder == None:
            folder=self.latest_outcmaes_folder_name()
        
        with open(folder+'qpc.pkl','rb') as file:
            QPC=pickle.load(file)
        return QPC


    
if __name__=='__main__':
    dat=datahandler()
    folder_name,run_id=dat.next_outcmaes_folder_name()
    print(folder_name)
    print(run_id)
# %%
