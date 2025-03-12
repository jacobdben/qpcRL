import cma
from os import cpu_count, listdir, mkdir
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

def save_qpca(qpca, folder):
    with open(folder+'saved_qpca.pkl','wb') as file:  # Overwrites any existing file.
        pickle.dump(qpca, file, pickle.HIGHEST_PROTOCOL)

def load_qpca(folder):
    qpca = None
    with open(folder+'saved_qpca.pkl','rb') as file:
        qpca = pickle.load(file)
    return qpca
            
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
    
    def load(self, folder):
        datadict = None
        with open(folder+'datadict.txt','rb') as file:
            datadict=json.load(file)
        return datadict


def parallel_cma(func_to_minimize,function_args, starting_point, sigma=0.5,options=None):

    
    savefolder = 'outcmaes/'

    if 'outcmaes' in listdir():
        nruns = len(listdir('outcmaes/'))
        if nruns > 0:
            savefolder += 'run_' + str(nruns+1) +'/'
        else:
            savefolder += 'run_1/'
        mkdir(savefolder)
    else:
        savefolder += 'run_1/'
        mkdir('outcmaes/')
        mkdir(savefolder)


    par_func_to_minimize=partial(func_to_minimize,**function_args)
    
    #start a datadict and measure the starting point, cma-es for some reason doesnt measure the starting point
    cmaesdata = CmaesData()
    starting_results = par_func_to_minimize(starting_point)[0]
    print("Unoptimised score:", starting_results)
    cmaesdata.add(0, starting_point, starting_results)

    if options == None:
        options={'timeout':24*60*60,'popsize':cpu_count()}

    options['verb_filenameprefix'] = savefolder
    
    
    es=cma.CMAEvolutionStrategy(starting_point,sigma,options)
    es.logger.disp_header()
    num_cpus=options['popsize'] 
    
    
    iteration=1
    while not es.stop(ignore_list=['tolfun']):
        solutions=es.ask()
        
        with ProcessPoolExecutor(cpu_count()) as executor:
            results = list(executor.map(par_func_to_minimize, solutions))
            results = [result[0] for result in results]
            results = np.array(results).astype(float)

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
    
    #save the qpca instance
    save_qpca(function_args['qpca'], savefolder)
    
    #save the datadict
    cmaesdata.save(savefolder)

    return es.best, es

