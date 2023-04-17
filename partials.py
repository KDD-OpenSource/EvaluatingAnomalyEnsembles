import numpy as np
import time


def _partials(q):
    if len(q) ==0:
        yield []
    else:
        for zw in _partials(q[1:]):
            yield [q[0]] + zw
            yield zw


def partials(lis, n_min=-1, n_max=-1):
    if n_max<0:
        n_max = len(lis)
    for q in _partials(lis):
        if n_min <= len(q) <= n_max:
            yield q


def _numshuffled(n=25,n_min=-1,n_max=-1):
    if n_max<0:n_max=n
    base=np.zeros((1,n),dtype="bool")
    ret=base
    for i in range(n):
        add=np.copy(ret)
        add[:,i]=1
        ret=np.concatenate((ret,add),axis=0)
        #print(ret.shape)
    seed=np.random.randint(0,100000)
    np.random.seed(12)
    np.random.shuffle(ret)
    np.random.seed(seed)
    #print(ret.shape)
    for val in ret:
        #cou=np.sum(val)
        #if cou>n_max or cou<n_min:continue
        yield val

def shuffled(lis,*args,**kwargs):
    lis=np.array(lis)
    #dex=[i for i in range(len(lis))]
    opts=_numshuffled(len(lis),*args,**kwargs)
    for opt in opts:
        #print(opt)
        yield lis[np.where(opt)]


if __name__=="__main__":
    lis = [i for i in range(21)]
    t0=time.time()
    #print(len(list(partials(lis, 5, -1))))
    #print(time.time()-t0)
    #exit()
    qq=[]
    from tqdm import tqdm
    for i,q in tqdm(enumerate(shuffled(lis))):
        pass
        #print(q)
        for zw in q:qq.append(zw)
        if i>1000:break
    




