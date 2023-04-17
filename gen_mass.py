import numpy as np
from sklearn.metrics import roc_auc_score

from datasets import iterate_data
from models import models

import os

from time import time
import func_timeout

import sys

def _fit_one(model, x,gx, tx):
    model.fit(x)
    return model.decision_function(tx), model.decision_function(gx)

dtmax=60

def fit_one(model,x,gx,tx):
    try:
        return func_timeout.func_timeout(dtmax, _fit_one, args=(model, x,gx, tx))
    except func_timeout.exceptions.FunctionTimedOut:
        print("Timeout")
        return f"Timeout {dtmax} seconds",""
    except Exception as e:
        print("Error",e)
        return f"Error {e}",""

def modp(p,p0):
    #pm=np.copy(p0)
    #quant=0.025
    #b1=np.quantile(pm,1-quant)
    #b2=np.quantile(pm,quant)
    #pm[np.where(pm>b1)]=b1
    #pm[np.where(pm<b2)]=b2
    #if np.std(pm)<np.std(p0)/1.5:
    #    p0=pm
    #    p[np.where(p>b1)]=b1
    #    
    mp=np.mean(p0)
    p/=mp
    p0/=mp
    return p,p0

ds=None
mdel=None
if len(sys.argv)>1:
    ds=sys.argv[1]
if len(sys.argv)>2:
    mdel=sys.argv[2]


for d,x,gx,tx,ty in iterate_data():
    if not ds is None and ds!=str(d):continue
    #if not str(d)=="cardio":continue
    print('Dataset:', str(d))
    bpth=f"results/{str(d)}/"
    os.makedirs(bpth, exist_ok=True)
    for nam,m in models.items():
        if mdel=="two":
            if nam=="ecod":pass
            elif nam=="copod":pass
            else:continue
        else:
            if not mdel is None and nam!=mdel:continue
        if nam=="kde" and len(x)>100000:continue
        print('Model:', nam)
        #if not nam=="ifor":continue
        pth=f"{bpth}{nam}.npz"
        #if os.path.exists(pth):
        #    continue
        model = m()
        t0=time()
        p,p0=fit_one(model, x,gx, tx)
        t1=time()
        if type(p)==str:
            auc="NA"
        else:    
            try:
                p,p0=modp(p,p0)
                auc=roc_auc_score(ty, p)
                if np.any(np.isnan(p)) or np.any(np.isnan(p0)):raise Exception("found nan")
            except:
                auc="NA"
        print('AUC:', auc,np.max(p))
        print()
        np.savez(pth, p=p, p0=p0, auc=auc,dt=t1-t0)
        if nam=="kde":exit()


exit()

condition=name=="cardio"

for d,x,tx,ty in pipeline(condition, split, shuffle, nonconst):
    print(d,x,tx,ty)




