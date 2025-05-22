import json as js
import numpy as np
import torch as th
from pathlib import Path
import pathlib
from datetime import datetime
"""
Mean square error
"""
def MSE(y,u):
    return (y-u).pow(2).mean()

"""
L2-Loss
"""

def L2(y,u):
    return (y-u).pow(2).sum().sqrt()


"""
Chose nonlinear
"""
def nonlin_choice(what):
    match what:
        case 'tanh':
            return th.tanh
        case 'sigmoid':
            return th.sigmoid
        case 'relu':
            return th.relu
        case _:
            return th.nn.functional.linear
        
"""
Choose optimizer
"""
def choose_optimizer(model,args):
    print(f"Optimizer choosen as {args.opti}")
    match args.opti:
        case 'Adadelta':
            print(f"Optimizer {args.opti} initiated")
            return th.optim.Adadelta(model.parameters(),args.learn_rate,weight_decay=args.weight_decay)
        case 'Adagrad':
            print(f"Optimizer {args.opti} initiated")
            return th.optim.Adagrad(model.parameters(),args.learn_rate,weight_decay=args.weight_decay)
        case 'Adam':
            print(f"Optimizer {args.opti} initiated")
            return th.optim.Adam(model.parameters(),args.learn_rate,weight_decay=args.weight_decay)
        case 'SparseAdam':
            print(f"Optimizer {args.opti} initiated")
            return th.optim.SparseAdam(model.parameters(),args.learn_rate,weight_decay=args.weight_decay)
        case 'Adamax':
            print(f"Optimizer {args.opti} initiated")
            return th.optim.Adamax(model.parameters(),args.learn_rate,weight_decay=args.weight_decay)
        case 'ASGD':
            print(f"Optimizer {args.opti} initiated")
            return th.optim.ASGD(model.parameters(),args.learn_rate,weight_decay=args.weight_decay)
        case 'LBFGS':
            print(f"Optimizer {args.opti} initiated")
            return th.optim.LBFGS(model.parameters(),args.learn_rate,weight_decay=args.weight_decay)
        case 'NAdam':
            print(f"Optimizer {args.opti} initiated")
            return th.optim.NAdam(model.parameters(),args.learn_rate,weight_decay=args.weight_decay)
        case 'RAdam':
            print(f"Optimizer {args.opti} initiated")
            return th.optim.RAdam(model.parameters(),args.learn_rate,weight_decay=args.weight_decay)
        case 'RMSprop':
            print(f"Optimizer {args.opti} initiated")
            return th.optim.RMSprop(model.parameters(),args.learn_rate,weight_decay=args.weight_decay)
        case 'SGD':
            print(f"Optimizer {args.opti} initiated")
            return th.optim.SGD(model.parameters(),args.learn_rate,weight_decay=args.weight_decay)
        case 'AdamW':
            print(f"Optimizer {args.opti} initiated")
            return th.optim.AdamW(model.parameters(),args.learn_rate,weight_decay=args.weight_decay)
        case _:
            print(f"Optimizer {args.opti} not recognized. Initiating AdamW instead.")
            return th.optim.AdamW(model.parameters(),args.learn_rate,weight_decay=args.weight_decay)


    


"""
Lagre data hvis man ønsker
"""

class NpEncoder(js.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj,pathlib.WindowsPath):
            return obj.__str__()
            
        return super().default(obj)


def savedatadict(data,name,wanted_placement_path:Path):
    where = wanted_placement_path/f"{name}.json"
    print(f"Saving to {where}")
    with where.open('w') as file:
        js.dump(data,file,sort_keys=True,indent=4,cls=NpEncoder) 
    print("Save finished") 



"""
Laste inn daten fra fila
"""

def loaddatadict(name, location:Path):
    dt ={}
    where = location/f"{name}.json"
    with where.open('r') as file:
        data = js.load(file)
        for i in data:
            dt[i] = np.asarray(data[i])
    return dt


def Get_time():
    return datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')


def Save_model(model:th.nn.Module,name,location:Path):
    where = location/f"{name}.tar"
    th.save(model.state_dict(), where)



def text_export(info,name,location):
    with open(location/f"{name}.txt","a") as f:
        f.write(str(info))


