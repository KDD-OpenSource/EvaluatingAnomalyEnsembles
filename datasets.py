import numpy as np

from yano.iter import *
from yano.symbols import *

#for d in pipeline(name):
#    print(d)

#print(len(name))
#exit()

def normal_split(q):
    cut=0.8
    cut=int(len(q)*cut)
    return q[:cut],q[cut:]

def normalize(d,x,gx,tx,ty):
    mn,std=np.mean(x,axis=0),np.std(x,axis=0)
    std+=1e-10
    x=(x-mn)/std
    gx=(gx-mn)/std
    tx=(tx-mn)/std

    return d,x,gx,tx,ty    

def iterate_data():
    for d,x,tx,ty in pipeline(name,split,shuffle,nonconst,normalize_zscore):
        a,b=normal_split(x)

        #yield normalize(d,a,b,tx,ty)
        yield d,a,b,tx,ty

def find_ty(nam):
    for d,x,tx,ty in pipeline(name==nam,split,shuffle,nonconst,normalize_zscore):
        return ty
    raise Exception(f"No such name {nam}")

if __name__=="__main__":
    for d,x,tx,ty in pipeline(name=="cover",split,shuffle,nonconst,normalize_zscore):
        print(d,x.shape,tx.shape,ty.shape)
        break
    exit()
    for d,x,tx,ty in iterate_data():
        print(d,x.shape,tx.shape,ty.shape)




