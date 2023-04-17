import numpy as np

from load import load_folder, list_folders
from partials import partials,shuffled
from combinations import combinations, whats
from datasets import find_ty
from basic_metrics import basics, thans
from modulo import modulo_name
from outputs import all_funcs

from tqdm import tqdm
from uqdm import uqdm

from sklearn.metrics import roc_auc_score
import json

import os

#lis = [label, scores, auc]

mincou=5

funcs=all_funcs(whats,thans)
keys=list(funcs.keys())

#print(values)
#exit()

modulo_max=1500
modulo=0
import sys
if len(sys.argv)>1:
    modulo=int(sys.argv[1])


def test_list_combination(lis,lislearn,ty, comb):
    global merge
    lis=lis.T
    lislearn=lislearn.T
    merge=comb(lis,lislearn)
    if np.any(np.isnan(merge)):
        return -1.0
    auc=roc_auc_score(ty,merge)
    return auc

def test_list(lis, lislearn, ty, combs=combinations):
    return {key: test_list_combination(lis,lislearn,ty,val) for key,val in combs.items()}

def basics_list(lis):
    return {key: val(lis) for key,val in basics.items()}

def test_dataset(dataset):
    values={key:0.0 for key in funcs.keys()}
    count=0
    #global count,keys,values,funcs
    data = load_folder(dataset)
    ty=find_ty(dataset)

    names=[zw[0] for zw in data]
    scores=[zw[1] for zw in data]
    scorels=[zw[2] for zw in data]
    aucs=[zw[3] for zw in data]

    values_x1={name:{key:0.0 for key in funcs.keys()} for name in names}
    values_x2={name1:{name2:{key:0.0 for key in funcs.keys()} for name2 in names} for name1 in names}
    count_x1={name:0 for name in names}
    count_x2={name1:{name2:0 for name2 in names} for name1 in names}

    for (i, (nam, score, scorel, auc)),func in uqdm(enumerate(zip(shuffled(names,mincou), shuffled(scores,mincou), shuffled(scorels,mincou), shuffled(aucs,mincou)))):
        if not (i%modulo_max==modulo):
            continue
        #print(len(nam))
        try:
        #for kkk in range(1):
            merges= test_list(np.array(score),np.array(scorel), ty)
            for key,val in merges.items():
                if val<0:raise Exception("nan"+str(key))
        except Exception as e:
            print(e)
            #exit()
            continue
        basics= basics_list(aucs)
        res={}
        res["merge"]=merges
        res["stats"]=basics
        #res['algos']=nam
        #res['dataset']=dataset


        #x0
        for key in keys:
            values[key]+=funcs[key](res)
        count+=1

        #x1
        for name in nam:
            for key in keys:
                values_x1[name][key]+=funcs[key](res)
            count_x1[name]+=1

        #x2
        for name1 in nam:
            for name2 in nam:
                for key in keys:
                    values_x2[name1][name2][key]+=funcs[key](res)
                count_x2[name1][name2]+=1


        func(count)

        if not count%100:
            print(json.dumps(values,indent=2))

    
        #fnn="finals/"+dataset+"/"+modulo_name(i,moduloid)+".json"
        #fon=fnn[:fnn.rfind('/')]
        #os.makedirs(fon, exist_ok=True)
        #with open(fnn, 'w') as f:
        #    json.dump(res, f, indent=2)
    return values,count,values_x1,count_x1,values_x2,count_x2

def dic_with(dic,key,val,key2=None,val2=None):
    ret={key:val}
    if not (key2 is None):
        ret[key2]=val2
    for k,v in dic.items():
        ret[k]=v
    return ret
 
def main():
    values={key:0.0 for key in funcs.keys()}
    count=0
    values_x1={}
    count_x1={}
    values_x2={}
    count_x2={}
    for i,folder in tqdm(enumerate(list_folders(modulo))):
        try:
            vals,cou,vals1,cou1,vals2,cou2=test_dataset(folder)
            #x0
            for key in keys:
                values[key]+=vals[key]
            count+=cou

            #x1
            for name in vals1.keys():
                if not name in values_x1:
                    values_x1[name]={key:0.0 for key in funcs.keys()}
                    count_x1[name]=0
                for key in keys:
                    values_x1[name][key]+=vals1[name][key]
                count_x1[name]+=cou1[name]

            #x2
            for name1 in vals2.keys():
                if not name1 in values_x2:
                    values_x2[name1]={}
                    count_x2[name1]={}
                for name2 in vals2[name1].keys():
                    if not name2 in values_x2[name1]:
                        values_x2[name1][name2]={key:0.0 for key in funcs.keys()}
                        count_x2[name1][name2]=0
                    for key in keys:
                        values_x2[name1][name2][key]+=vals2[name1][name2][key]
                    count_x2[name1][name2]+=cou2[name1][name2]
            print(json.dumps({key:val/(count+1e-10) for key,val in values.items()},indent=2))
            with open(f"mid/{modulo}.json", 'w') as f:
                json.dump(dic_with(values,"count",count,"index",i), f, indent=2)
            with open(f"x1/{modulo}.json","w") as f:
                json.dump(dic_with(values_x1,"count",count_x1,"index",i), f, indent=2)
            with open(f"x2/{modulo}.json","w") as f:
                json.dump(dic_with(values_x2,"count",count_x2,"index",i), f, indent=2)
        except Exception as e:
            for jj in range(100):
                print("critical error",i,folder)
            print(e)

    values["count"]=count
    with open(f"kappa/{modulo}.json", 'w') as f:
        json.dump(values, f, indent=2)


main()


#test_dataset("cardio")




