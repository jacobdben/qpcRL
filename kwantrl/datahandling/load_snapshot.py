import ast

def load_snapshot(path):
    with open(path,'r') as file:
        cont=file.read()
        dictionary=ast.literal_eval(cont)
    
    values_dict={}
    for instrument in dictionary['station']['instruments'].keys():
        values_dict[instrument]={}
        for parameter in dictionary['station']['instruments'][instrument]['parameters'].keys():
            if 'value' in dictionary['station']['instruments'][instrument]['parameters'][parameter].keys():
                values_dict[instrument][parameter]=dictionary['station']['instruments'][instrument]['parameters'][parameter]['value']
    return values_dict

if __name__=='__main__':
    path='C:/Users/Torbjørn/Downloads/snapshot.dat'
    d=load_snapshot(path)
