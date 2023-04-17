import numpy as np



if __name__=="__main__":
    from yano.iter import *
    from yano.symbols import *

    condition=name=="cardio"
    
    for d,x,tx,ty in pipeline(condition, split, shuffle, nonconst,normalize_zscore):
        break


from pyod.models.knn import KNN
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.abod import ABOD
from pyod.models.mad import MAD
from pyod.models.kde import KDE
from pyod.models.gmm import GMM
from pyod.models.pca import PCA
from pyod.models.mcd import MCD
from pyod.models.mcd import MCD
from pyod.models.cd import CD
from pyod.models.ocsvm import OCSVM
from pyod.models.lmdd import LMDD
from pyod.models.lof import LOF
from pyod.models.cof import COF
from pyod.models.cblof import CBLOF
from pyod.models.loci import LOCI
from pyod.models.hbos import HBOS
from pyod.models.sod import SOD
from pyod.models.rod import ROD
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
#from pyod.models.feature_bagging import FeatureBagging
from pyod.models.lscp import LSCP
from pyod.models.loda import LODA

from pyod.models.anogan import AnoGAN
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.lunar import LUNAR
from pyod.models.vae import VAE


class AEplus():
    def __init__(self, *args, **kwargs):
        self.args,self.kwargs=args,kwargs

    def fit(self,X):
        dim=X.shape[1]
        self.ae=AutoEncoder(hidden_neurons=[dim,dim//2,dim//2,dim],*self.args,**self.kwargs)
        self.ae.fit(X)
    def decision_function(self,X):
        return self.ae.decision_function(X)

class VAEplus():
    def __init__(self, *args, **kwargs):
        self.args,self.kwargs=args,kwargs

    def fit(self,X):
        dim=X.shape[1]
        self.ae=VAE(encoder_neurons=[dim,dim//2,dim//3],decoder_neurons=[dim//3,dim//2,dim],*self.args,**self.kwargs)
        self.ae.fit(X)
    def decision_function(self,X):
        return self.ae.decision_function(X)




models={"knn1":lambda: KNN(n_neighbors=1),"knn5":lambda: KNN(n_neighbors=5),"knn_mean":lambda: KNN(method="mean"),"knn_median":lambda: KNN(method="median"),"ecod":ECOD,"copod":COPOD,"abod":ABOD,"kde":KDE,"gmm":GMM,"pca":PCA,"ocsvm":OCSVM,"lmdd":LMDD,"lof":LOF,"COF":COF,"cblof":CBLOF,"hbos":HBOS,"sod":SOD,"ifor":IForest,"inne":INNE,"loda":LODA}

new_models={"ae":AEplus,"deepsvdd":DeepSVDD,"lunar":LUNAR,"vae":VAEplus,"anogan":AnoGAN}

models.update(new_models)

if __name__=="__main__" and False:
    from sklearn.metrics import roc_auc_score
    clf=IForest()#LODA()
    clf.fit(x)
    p=clf.decision_function(tx)
    auc=roc_auc_score(ty,p)
    print(auc)
    exit()

if __name__=="__main__":
    import sys
    if not "full" in sys.argv:
        sys.exit(0)

    print(len(list(models.keys())))

    for nam,model in new_models.items():
        clf=model()
        clf.fit(x)
        p=clf.decision_function(tx)
        auc=roc_auc_score(ty,p)
        print(nam,auc)








