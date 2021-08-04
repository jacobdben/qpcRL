import cma

# general
import numpy as np
import os
import json
import pickle

def folder_name(data_path):
    if not os.path.exists(data_path+"outcmaes"):
        os.mkdir(data_path+"outcmaes")
    folders=folders=list(os.walk(data_path+"outcmaes/"))[0][1]
    lis=[int(f) for f in folders]
    lis.append(0)
    newfolder=data_path+'outcmaes/'+'{}/'.format(max(lis)+1)
    return newfolder


def optimize_cma(func_to_minimize,datahandler,start_point,maxfevals=99999,sigma=0.5,stop_time=None,callbacks=[None],args=[],options={}):
    if not stop_time==None:
        def callback_time(es):
            try:
                cur_time=es.time_last_displayed
            except AttributeError:
                return
            if cur_time>=stop_time:
                es.stop()['time']=cur_time
        
        if callbacks==None:
            callbacks=[callback_time]
        else:
            callbacks.append(callback_time)
        
    
    #make a seperate folder for this run
    data_path=datahandler.data_path
    newfolder=folder_name(data_path)
    print("data saved to:")
    print(newfolder)
    os.mkdir(newfolder[:-1])
    
    #start a datadict and measure the starting point, cma-es for some reason doesnt measure the starting point
    datadict={'next_key':0,'measurements':{},'starting_point':{'next_key':0,'measurements':{}}}
    func_to_minimize(start_point,datadict['starting_point'])
    
    args_send=[datadict]
    args_send.extend(args)


    options_send={'maxfevals':maxfevals,'verb_filenameprefix':newfolder}
    for key in options:
        options_send[key]=options[key]
        
    x,es=cma.fmin2(func_to_minimize,start_point,sigma0=sigma,args=args_send,options=options_send,callback=callbacks)

    #save stopping criterion
    with open(newfolder+"stopping_criterion.txt",mode='w') as file_object:
        print(es.stop(),file=file_object)
    
    #save the es instance
    string=es.pickle_dumps()
    with open(newfolder+'saved_es.pkl','wb') as file:
        file.write(string)
    
    #save the datadict
    with open(newfolder+"datadict.txt",mode='w') as file_object:
        file_object.write(json.dumps(datadict))
        
    return x,es, int(newfolder[-2:-1])


def resume_cma(func_to_minimize,run_id,datahandler,maxfevals=99999,stop_time=None,callbacks=[None],args=[],options={}):
    data_path=datahandler.data_path
    folder=data_path+'outcmaes/{}/'.format(run_id)

    print("data loaded from:")
    print(folder)
    
    with open(folder+'datadict.txt','rb') as file:
        datadict=json.load(file)
        
    with open(folder+'saved_es.pkl','rb') as file:
        string=file.read()
        es=pickle.loads(string)

    if not stop_time==None:
        start_time=es.time_last_displayed
        def callback_time(es):
            cur_time=es.time_last_displayed
            if cur_time>=start_time+stop_time:
                es.stop()['time']=cur_time
        
        callbacks.append(callback_time)
    args_send=[datadict]
    args_send.extend(args)

    options_send={'maxfevals':maxfevals}
    # for key in options:
    es.opts.set(options) # this change is untested so far
    es.opts.set(options_send )
        
    es.optimize(func_to_minimize,args=[datadict],callback=[callback_time])#,options=options_send) see comment above
    
    with open(folder+"stopping_criterion.txt",mode='a+') as file_object:
        print(es.stop(),file=file_object)
        

    
    with open(folder+"datadict.txt",mode='w') as file_object:
        file_object.write(json.dumps(datadict))
        
    string=es.pickle_dumps()
    with open(folder+'saved_es.pkl','wb') as file:
        file.write(string)
        
    return es.result.xbest, es, run_id
