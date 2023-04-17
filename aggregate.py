import numpy as np
import os


def aggregate_file(fn):
    try:
        f=np.load(fn)
        q=f["p"].tolist()
        #print(type(q))
        if type(q)==str:
            return 0
        else:
            #return bool(float(f["auc"])>0.5)
            return 1 if not np.any(np.isnan(q)) else 0
            return 1

    except:
        return 0

def aggregate_dir(dn):
    files=os.listdir(dn)
    files=[dn+"/"+fn for fn in files]
    return sum([aggregate_file(fn) for fn in files])

def aggregate_root():
    dirs=os.listdir("results/")
    dfns=["results/"+dn for dn in dirs]
    count=0
    for d,fn in zip(dirs,dfns):
        ac=aggregate_dir(fn)
        print(d,ac)
        count+=ac
    print("Total:",count)

if __name__=="__main__":
    aggregate_root()



