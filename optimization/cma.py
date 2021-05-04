import cma

# general
import numpy as np
import os
import time
import json

def folder_name(data_path):
    if not os.path.exists(data_path+"outcmaes"):
        os.mkdir(data_path+"outcmaes")
    folders=folders=list(os.walk(data_path+"outcmaes/"))[0][1]
    lis=[int(f) for f in folders]
    lis.append(0)
    newfolder=data_path+'outcmaes/'+'{}/'.format(max(lis)+1)
    return newfolder


def optimize_cma(func_to_minimize,datahandler,maxfevals,sigma=0.5,start_point=np.zeros(2),time_stop=None,callbacks=[None],args=[]):
    global global_start_time, global_stop_time
    if not time_stop==None:
        global_start_time=time.time()
        print(global_start_time)
        global_stop_time=time_stop
        print(global_stop_time)
        def callback_time(es):
            global global_start_time, global_stop_time
            cur_time=time.time()
            # print('cur_time %f'%cur_time)
            if (cur_time-global_start_time)>global_stop_time:
                print('STOP')
                print(cur_time)
                print(global_start_time)
                print(global_stop_time)
                es.stop()['time']=(cur_time-global_start_time) #{'start_time':start_time,'cur_time':cur_time,'stop_time':stop_time}
        if callbacks==None:
            callbacks=[callback_time]
        else:
            callbacks.append(callback_time)
        
    
    data_path=datahandler.data_path
    newfolder=folder_name(data_path)
    print("data saved to:")
    print(newfolder)
    os.mkdir(newfolder[:-1])
    dataidsdict={}
    args_send=[dataidsdict]
    args_send.extend(args)
    x,es=cma.fmin2(func_to_minimize,start_point,sigma0=sigma,args=args_send,options={'maxfevals':maxfevals,'verb_filenameprefix':newfolder},callback=callbacks)
    with open(newfolder+"stopping_criterion.txt",mode='w') as file_object:
        print(es.stop(),file=file_object)
        
    # print(dataidsdict)
    with open(newfolder+"dataids.txt",mode='w') as file_object:
        file_object.write(json.dumps(dataidsdict))
        
    return x,es



# plot=False
# if plot:
#     x,p=new_point(x,bounds=(-0.5,0.5))
    
#     print("result had penalty: {}".format(p))
#     print(x)
#     result=[]
#     baseline=[]
#     plt.figure()
#     for avg_gates in common_voltages:
#         QPC.set_all_pixels(x+avg_gates)
#         result.append(QPC.transmission())
#     plt.plot(common_voltages,result,label='Optimized:{:.3f}'.format(stairs.histogram(result)))

    
#     for avg_gates in common_voltages:
#         QPC.set_all_pixels(avg_gates)
#         baseline.append(QPC.transmission())
#     plt.plot(common_voltages,baseline,label='Non Optimized:{:.3f}'.format(stairs.histogram(baseline)))

#     plt.xlabel('Avg Voltage [V]')
#     plt.ylabel('Conductance')
#     plt.grid()
#     plt.legend()
    
    
#     bounds=((19,41),(25,45))
#     fig,ax=plt.subplots() 
#     QPC.U0=disorder
#     ax.set_title("Optimized") 
#     QPC.set_all_pixels(x+common_voltages[0])
#     t1,p1=QPC.plot_potential_section(bounds=bounds,ax=ax)
#     plt.colorbar(p1)
    
#     fig,ax=plt.subplots()
#     ax.set_title("Non Optimized")
#     QPC.U0=disorder
#     QPC.set_all_pixels(common_voltages[0])
#     t2,p2=QPC.plot_potential_section(bounds=bounds,ax=ax)
#     plt.colorbar(p2)
#     np.where(t1==t2)
    
#     fig,ax=plt.subplots()
#     ax.set_title("Non Optimized,no disorder")
#     QPC.U0=0
#     QPC.set_all_pixels(common_voltages[0])
#     t2,p2=QPC.plot_potential_section(bounds=bounds,ax=ax)
#     plt.colorbar(p2)
#     np.where(t1==t2)
