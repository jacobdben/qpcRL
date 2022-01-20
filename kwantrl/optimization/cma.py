import cma

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

