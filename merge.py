import json
import os

fold="mid/"
fns=[fold+zw for zw in os.listdir(fold)]

def loadone(fn):
    with open(fn,"r") as f:
        stri=f.read()
        dic=json.loads(stri)
    return dic

dics=[loadone(fn) for fn in fns]

keys=dics[0].keys()

total={key:0 for key in keys}
for dic in dics:
    for key in keys:
        total[key]+=dic[key]

what="_"
what="bigger"
#what="average"
#what="aucdistance"

ending=""
ending="max"
ending="mean"
#ending="min"

frac={key:val/total["count"] for key,val in total.items() if key.startswith(what) and key.endswith(ending)}
frac["count"]=total["count"]
frac["index"]=total["index"]




print(json.dumps(frac,indent=2))


