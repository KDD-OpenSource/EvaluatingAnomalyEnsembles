import os
import numpy as np


def load_file(fn):
    f=np.load(fn)
    #p, auc
    p=f['p']
    pl=f['p0']
    auc=f['auc']
    if type(p.tolist()) is str:
        raise Exception('no value here')
    if np.any(np.isnan(p)) or np.any(np.isnan(pl)):
        raise Exception('nan')
    return p,pl,auc

def cut_task(fn):
    fn=fn[fn.rfind('/')+1:]
    fn=fn[:fn.rfind('.')]
    return fn

def load_folder(fold):
    fns=os.listdir("results/"+fold)
    fns=[[cut_task(fn),os.path.join("results/"+fold,fn)] for fn in fns]

    q=[]
    for x,y in fns:
        try:
            a,b,c=load_file(y)
        except Exception as e:
            continue
        q.append([x,a,b,c])
    
    #q=[[x,*load_file(y)] for x,y in fns]
    return q

def list_folders(equate=0):
    lis=os.listdir("results")
    lis.sort()
    seed=np.random.randint(0,100000)
    np.random.seed(equate)
    np.random.shuffle(lis)
    np.random.seed(seed)
    return lis
    #return os.listdir('results')

    




