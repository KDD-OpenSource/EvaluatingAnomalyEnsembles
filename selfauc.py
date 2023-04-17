from yano.iter import *
from yano.symbols import *

import numpy as np
from plt import *
import json
import os

from sklearn.metrics import roc_auc_score
from tqdm import tqdm

tys={}
def findty(ds):
    global tys
    if ds in tys:
        return tys[ds]
    else:
        for d,x,tx,ty in pipeline(name==ds,split,shuffle):
            break
        tys[ds]=ty
        return ty


pth0="results/"
datasets=os.listdir(pth0)

def doone(dataset):
    pth=pth0+dataset+"/"
    ty=findty(dataset)
    algos=os.listdir(pth)
    dic={}
    for algo in algos:
        nam=algo[:-4]
        f=np.load(f"{pth}{algo}")
        try:
            p=f["p"]
            if len(p)==0:continue
            auc=roc_auc_score(ty,p)
            dic[nam]=auc
        except:
            pass
    return dic


dics={dataset:doone(dataset) for dataset in tqdm(datasets)}
with open("mediate/pure_auc.json","w") as f:
    json.dump(dics,f,indent=2)


