import cma
from os import cpu_count
from concurrent.futures import ProcessPoolExecutor

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


            
class CmaesData():
    def __init__(self):
        self.data = []
    
    def add(self, iteration, coordinate, loss):
        
        if type(coordinate) is np.ndarray:
            coordinate = list(coordinate.flatten())
        
        self.data.append({'iteration': iteration,  'coordinate': coordinate, 'loss': loss})
    
    def save(self, folder):
        with open(folder+"datadict.txt",mode='w') as file_object:
            file_object.write(json.dumps(self.data))


def parallel_cma(func_to_minimize,function_args, starting_point, sigma=0.5,options=None):

    
    savefolder = 'outcmaes/'
    

    par_func_to_minimize=partial(func_to_minimize,**function_args)
    
    #start a datadict and measure the starting point, cma-es for some reason doesnt measure the starting point
    cmaesdata = CmaesData()
    starting_results = par_func_to_minimize(starting_point)
    cmaesdata.add(0, starting_point, starting_results)

    if options == None:
        options={'timeout':2,'popsize':cpu_count()}

    options['verb_filenameprefix'] = savefolder
    
    
    es=cma.CMAEvolutionStrategy(starting_point,sigma,options)
    es.logger.disp_header()
    num_cpus=options['popsize'] 
    
    
    iteration=1
    while not es.stop():
        solutions=es.ask()
        
        with ProcessPoolExecutor(cpu_count()) as executor:
            results = np.array(list(executor.map(par_func_to_minimize, solutions))).astype(float)
            print(results)
            print(solutions)
            for i in range(len(results)):
                cmaesdata.add(iteration, solutions[i], results[i])
            

        es.tell(solutions,results)
        es.logger.add()
        es.disp()
        iteration+=1
    es.result_pretty()[0][0]

    with open(savefolder+"stopping_criterion.txt",mode='w') as file_object:
        print(es.stop(),file=file_object)
    
    #save the es instance
    save_es(es, savefolder)
    
    #save the datadict
    cmaesdata.save(savefolder)

    return es.best, es

