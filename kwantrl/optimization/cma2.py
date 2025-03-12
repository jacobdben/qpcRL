import cma
from multiprocessing import Pool, cpu_count
from kwantrl.datahandling.datahandling import datahandler as dat_class
from functools import partial

# general
import numpy as np
import os
import json
import pickle

def save_es(es,folder):
    string=es.pickle_dumps()
    with open(folder+'saved_es.pkl','wb') as file:
        file.write(string)

def load_es(folder):
    with open(folder+'saved_es.pkl','rb') as file:
        string=file.read()
        es=pickle.loads(string)
    return es

def optimize_cma(func_to_minimize,datahandler,start_point,maxfevals=99999,sigma=0.5,stop_time=None,callbacks=[None],args=[],options={},QPC=None):
    #make a seperate folder for this run
    newfolder,run_id=datahandler.next_outcmaes_folder_name()
    print("data saved to:")
    print(newfolder)
    os.mkdir(newfolder[:-1])
    if QPC!=None:
        datahandler.save_qpc(QPC,run_id)
    
    #start a datadict and measure the starting point, cma-es for some reason doesnt measure the starting point
    datadict={'next_key':0,'measurements':{},'starting_point':{'next_key':0,'measurements':{}}}
    func_to_minimize(start_point,datadict['starting_point'])
    
    args_send=[datadict]
    args_send.extend(args)


    options_send={'maxfevals':maxfevals,'verb_filenameprefix':newfolder}
    for key in options:
        options_send[key]=options[key]
    if not stop_time==None:
        options_send['timeout']=stop_time
        
    x,es=cma.fmin2(func_to_minimize,start_point,sigma0=sigma,args=args_send,options=options_send,callback=callbacks)

    #save stopping criterion
    with open(newfolder+"stopping_criterion.txt",mode='w') as file_object:
        print(es.stop(),file=file_object)
    
    #save the es instance
    save_es(es,newfolder)
    
    #save the datadict
    datahandler.save_data(datadict,newfolder)
        
    return x,es, run_id


def cma_p(func_to_minimize,function_args,starting_point=np.zeros(9),sigma=0.5,datahandler=None,QPC=None,options=None):

    if datahandler==None:
        datahandler=dat_class()
    #make a seperate folder for this run
    newfolder,run_id=datahandler.next_outcmaes_folder_name()
    print("data saved to:")
    print(newfolder)
    os.mkdir(newfolder[:-1])
    if QPC!=None:
        datahandler.save_qpc(QPC,run_id)
    
    #start a datadict and measure the starting point, cma-es for some reason doesnt measure the starting point
    datadict={'measurements':{},'starting_point':{'next_key':0,'measurements':{}}}
    starting_results=func_to_minimize(starting_point,**function_args)
    datadict['starting_point']=starting_results[1]

    if options == None:
        options={'timeout':100,'popsize':cpu_count()}

    options['verb_filenameprefix']=newfolder
    
    
    es=cma.CMAEvolutionStrategy(starting_point,sigma,options)
    es.logger.disp_header()
    num_cpus=options['popsize'] 
    
    par_func_to_minimize=partial(func_to_minimize,**function_args)
    iteration=1
    while not es.stop():
        solutions=es.ask()
        datadict['measurements'][str(iteration)]={}
        with Pool(num_cpus) as p:
            result_list=p.map(par_func_to_minimize,solutions)
        # return result_list
            result_list=np.array(result_list)
            result=result_list[:,0].astype(float)
            for index,data_to_save in enumerate(result_list[:,1]):
                datadict['measurements'][str(iteration)][str(index)]=data_to_save
            

        es.tell(solutions,result)
        es.logger.add()
        es.disp()
        iteration+=1
    es.result_pretty()[0][0]

    with open(newfolder+"stopping_criterion.txt",mode='w') as file_object:
        print(es.stop(),file=file_object)
    
    #save the es instance
    save_es(es,newfolder)
    
    #save the datadict
    datahandler.save_data(datadict,newfolder)

    return es.best, es, run_id


def resume_cma(func_to_minimize,run_id,datahandler,maxfevals=99999,stop_time=None,callbacks=[None],args=[],options={}):
    data_path=datahandler.data_path
    folder=data_path+'outcmaes/{}/'.format(run_id)

    print("data loaded from:")
    print(folder)
    
    #load datadict
    datadict=datahandler.load_data(folder=folder)
    #load es
    es=load_es(folder)

    if not stop_time==None:
        start_time=es.time_last_displayed

    args_send=[datadict]
    args_send.extend(args)

    options_send={'maxfevals':maxfevals,'timeout':stop_time+start_time}
    # for key in options:
    es.opts.set(options) # this change is untested so far
    es.opts.set(options_send )
        
    es.optimize(func_to_minimize,args=[datadict])#,options=options_send) see comment above
    
    with open(folder+"stopping_criterion.txt",mode='a+') as file_object:
        print(es.stop(),file=file_object)
    
    datahandler.save_data(datadict,folder)
        
    save_es(es,folder)

    return es.result.xbest, es, run_id


class CmaRunner():
    def __init__(self,measurement_func,starting_point,constraint_func=lambda x: (x,0),sigma0=0.5,measurement_func_args=[],datahandler=None,QPC=None,cma_options={},run_id=None) -> None:
        if run_id!=None:
            raise Exception("continuing not implemented here yet")
        
        #possibly move things into optimize, there is no need to init all the stuff at first.
        self.measurement_func=partial(measurement_func,*measurement_func_args)
        self.constraint_func=constraint_func
        
        if datahandler==None:
            self.datahandler=dat_class()
        else:
            self.datahandler=datahandler
            
        #make a seperate folder for this run
        self.newfolder,self.run_id=self.datahandler.next_outcmaes_folder_name()
        print(f"data saved to:{self.newfolder}")
        os.mkdir(self.newfolder[:-1])
        if QPC!=None:
            self.datahandler.save_qpc(QPC,run_id)
        
        cma_options['verb_filenameprefix']=self.newfolder
        opts=cma.CMAOptions(**cma_options)
        self.es = cma.CMAEvolutionStrategy(starting_point,sigma0,opts)
        #self.es.opts.set(options)
        self.iter_nr=1
        
        self.datadict={'measurements':{},'starting_point':{}}
        self.datadict['starting_point']=self._one_measurement(starting_point,starting_point=True)
        
    def optimize(self,):
        while not self.es.stop():
            self._one_iteration()   
        
        self.es.result_pretty()
        
        with open(self.newfolder+"stopping_criterion.txt",mode='w') as file_object:
            print(self.es.stop(),file=file_object)
    
        #save the es instance
        save_es(self.es,self.newfolder)
        
        #save the datadict
        self.datahandler.save_data(self.datadict,self.newfolder)
    
    def _one_iteration(self,):
        self.pop_nr=1
        self.datadict['measurements'][str(self.iter_nr)]={}
        if not self.es.stop():
            proposed_solutions=self.es.ask()
            evaluated_solutions=[self._one_measurement(proposed_solution) for proposed_solution in proposed_solutions]
            self.es.tell(proposed_solutions,evaluated_solutions)
            self.es.logger.add()
            self.es.disp()
            self.pop_nr+=1
            self.iter_nr+=1        
        
    def _one_measurement(self,solution,starting_point=False):
        new_solution,penalty = self.constraint_func(solution)
        results_dict=self.measurement_func(new_solution)
        results_dict['penalty']=penalty
        self._add_to_dict(results_dict,starting_point)
        return results_dict['val']+penalty
        
    def _add_to_dict(self,results_dict,starting_point=False):
        if starting_point:
            self.datadict['starting_point']=results_dict
        else:
            self.datadict['measurements'][str(self.iter_nr)][str(self.pop_nr)]=results_dict