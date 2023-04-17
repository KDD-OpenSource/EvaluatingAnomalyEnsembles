import numpy as np
from scipy.stats import multivariate_normal



def wrappable(norm):
    def subfunc(func, *args0, **kwargs0):

        def wrapper(data,data0,*args, **kwargs):
            data,data0 = norm(data,data0,*args, **kwargs)
            return func(data,data0,*args, **kwargs)

        if callable(func):
            return wrapper
        else:
            return norm(func, *args0, **kwargs0)
    return subfunc

@wrappable
def normalisation_min_max(data, data0=None, calcon=1,*args, **kwargs):
    """Takes the second to last dimension of the data and assures that every value lies between 0 and 1"""
    if calcon==1:
        mn,mx=np.min(data,axis=-2,keepdims=True),np.max(data,axis=-2,keepdims=True)
    elif calcon==0:
        mn,mx=np.min(data0,axis=-2,keepdims=True),np.max(data0,axis=-2,keepdims=True)
    data = data - mn
    data = data / (mx - mn)
    data0 = data0 - mn
    data0 = data0 / (mx - mn)
    return data, data0

@wrappable
def normalisation_zscore(data, data0=None, calcon=0,*args, **kwargs):
    """Takes the second to last dimension of the data and assures that mean=0 and std=1"""
    if calcon==1:
        mn,std=np.mean(data,axis=-2,keepdims=True),np.std(data,axis=-2,keepdims=True)
    elif calcon==0:
        mn,std=np.mean(data0,axis=-2,keepdims=True),np.std(data0,axis=-2,keepdims=True)
    data = data - mn
    data = data / std
    data0 = data0 - mn
    data0 = data0 / std
    return data, data0

@wrappable
def normalisation_clipped_zscore(data, data0=None, calcon=0,zcut=1,*args, **kwargs):
    data,data0=normalisation_zscore(data,data0,calcon=calcon)
    data[data>zcut]=zcut
    data[data<-zcut]=-zcut
    data0[data0>zcut]=zcut
    data0[data0<-zcut]=-zcut
    return data,data0


@wrappable
def normalisation_clipped_hard_zscore(data, data0=None, calcon=0,zcut=2,*args, **kwargs):
    data,data0=normalisation_zscore(data,data0,calcon=calcon)
    data[data>zcut]=zcut
    data[data<-zcut]=-zcut
    data0[data0>zcut]=zcut
    data0[data0<-zcut]=-zcut
    return data,data0


def normalise(usual="min_max",usual_calcon=0):
    def _normalise(func):
        """Decorator that adds a variable normalisation argument to the function"""
        def wrapper(data, data0 ,*args,norm=usual, calcon=usual_calcon, **kwargs):
            if norm == "min_max" or norm == "minmax" or norm == "01":
                data, data0 = normalisation_min_max(data,data0,calcon=calcon)
            elif norm == "zscore" or norm == "z_score":
                data, data0 = normalisation_zscore(data,data0,calcon=calcon)
            elif norm =="none":
                mn=np.mean(data0)
                data/=mn
                data0/=mn
            elif norm=="clipped1":
                data,data0=normalisation_clipped_zscore(data,data0,calcon=calcon)
            elif norm=="clipped2":
                data,data0=normalisation_clipped_hard_zscore(data,data0,calcon=calcon)
            elif callable(norm):
                data = norm(data,calcon=calcon)
                data0 = norm(data0,calcon=calcon)
            else:
                pass
            return func(data,data0, *args, **kwargs)
        return wrapper
    return _normalise

def two_argument(func):
    """if given q and q0, do nothing, if not given q0 set it to q"""
    #print(func.__code__.co_varnames)
    def wrapper(q, q0=None, *args, **kwargs):
        if q0 is None:
            q0 = q
        return func(q, q0, *args, **kwargs)
    return wrapper


@two_argument
@normalise()
def maximum(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the maximum value for each sample."""
    return np.max(q, axis=-1)

@two_argument
@normalise("zscore")
def maximum_zscore(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the maximum value for each sample."""
    return np.max(q, axis=-1)

@two_argument
@normalise("none")
def maximum3(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the maximum value for each sample."""
    return np.max(q, axis=-1)

@two_argument
@normalise("clipped1")
def maximum4(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the maximum value for each sample."""
    return np.max(q, axis=-1)

@two_argument
@normalise("clipped2")
def maximum5(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the maximum value for each sample."""
    return np.max(q, axis=-1)




@two_argument
@normalise()
def minimum(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the minimum value for each sample."""
    return np.min(q, axis=-1)

@two_argument
@normalise("zscore")
def minimum_zscore(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the minimum value for each sample."""
    return np.min(q, axis=-1)

@two_argument
@normalise("none")
def minimum3(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the minimum value for each sample."""
    return np.min(q, axis=-1)

@two_argument
@normalise("clipped1")
def minimum4(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the minimum value for each sample."""
    return np.min(q, axis=-1)

@two_argument
@normalise("clipped2")
def minimum5(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the minimum value for each sample."""
    return np.min(q, axis=-1)


@two_argument
@normalise()
def mean(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the mean value for each sample."""
    return np.mean(q, axis=-1)

@two_argument
@normalise("zscore")
def mean_zscore(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the mean value for each sample."""
    return np.mean(q, axis=-1)

@two_argument
@normalise("none")
def mean3(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the mean value for each sample."""
    return np.mean(q, axis=-1)

@two_argument
@normalise("clipped1")
def mean4(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the mean value for each sample."""
    return np.mean(q, axis=-1)

@two_argument
@normalise("clipped2")
def mean5(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the mean value for each sample."""
    return np.mean(q, axis=-1)

@two_argument
@normalise("minmax")
def median(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the median value for each sample."""
    return np.median(q, axis=-1)

@two_argument
@normalise("zscore")
def median_beta(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the median value for each sample."""
    return np.median(q, axis=-1)

@two_argument
@normalise("none")
def median_gamma(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the median value for each sample."""
    return np.median(q, axis=-1)

@two_argument
@normalise("clipped1")
def median_delta(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the median value for each sample."""
    return np.median(q, axis=-1)

@two_argument
@normalise("clipped2")
def median_epsilon(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the median value for each sample."""
    return np.median(q, axis=-1)




def lnmean(n=2):
    """Generates means by a higher power n=2"""
    @two_argument
    @normalise()
    def func(q, q0):
        """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the mean value for each sample."""
        return (np.mean(np.abs(q)**n, axis=-1))**(1/n)
    return func

l2mean = lnmean(2)
l3mean = lnmean(3)
l4mean = lnmean(4)

def lnmean_zscore(n=2):
    """Generates means by a higher power n=2"""
    @two_argument
    @normalise("zscore")
    def func(q, q0):
        """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the mean value for each sample."""
        return (np.mean(np.abs(q)**n, axis=-1))**(1/n)
    return func

l2mean_zscore = lnmean_zscore(2)
l3mean_zscore = lnmean_zscore(3)
l4mean_zscore = lnmean_zscore(4)

def lnmean3(n=2):
    """Generates means by a higher power n=2"""
    @two_argument
    @normalise("none")
    def func(q, q0):
        """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the mean value for each sample."""
        return (np.mean(np.abs(q)**n, axis=-1))**(1/n)
    return func

l2mean3 = lnmean3(2)
l3mean3 = lnmean3(3)
l4mean3 = lnmean3(4)

def lnmean4(n=2):
    """Generates means by a higher power n=2"""
    @two_argument
    @normalise("clipped1")
    def func(q, q0):
        """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the mean value for each sample."""
        return (np.mean(np.abs(q)**n, axis=-1))**(1/n)
    return func

l2mean4 = lnmean4(2)
l3mean4 = lnmean4(3)
l4mean4 = lnmean4(4)

def lnmean5(n=2):
    """Generates means by a higher power n=2"""
    @two_argument
    @normalise("clipped2")
    def func(q, q0):
        """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the mean value for each sample."""
        return (np.mean(np.abs(q)**n, axis=-1))**(1/n)
    return func

l2mean5 = lnmean5(2)
l3mean5 = lnmean5(3)
l4mean5 = lnmean5(4)

def gen_knn(k=3,metric=2):
    """Generates a function that returns the k nearest neighbours distance to q0 of each sample in q. Effectively requires explicit q0 at least for k=1"""
    @two_argument
    @normalise("zscore")
    def func(q, q0):
        """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the k nearest neighbours distance to q0 of each sample."""
        #print("pras A",q.shape)
        #print("pras B",q0.shape)
        #print("pras C",(q[0] - q0).shape)
        #distances=np.array([np.sum(np.abs(qq-q0)**metric,axis=-1) for qq in q])
        #print("pras D",distances.shape)
        def takek(arr,k):
            if len(arr)<=k:return arr[-1]
            return arr[k]
        
        return np.array([takek(np.sort(np.sum(np.abs(qq - q0)**metric,axis=-1), axis=-1),k)**(1/metric) for qq in q])

    return func

knn1=gen_knn(1)
knn3=gen_knn(3)
knn5=gen_knn(5)
knn10=gen_knn(10)
knn100=gen_knn(100)



@two_argument
@normalise("minmax")
def gaussian_fit(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the gaussian probability under the assumption of q0"""
    mn=np.mean(q0,axis=-2)
    cov=np.cov(q0,rowvar=False)
    return -multivariate_normal.pdf(q,mean=mn,cov=cov)

from sklearn.ensemble import IsolationForest

@two_argument
@normalise("zscore")
def ifor_fit(q, q0):
    """Expects data of the shape (samples, algorithms). Normalises the algorithms and returns the ifor distance"""
    clf=IsolationForest().fit(q0)
    return -clf.decision_function(q)
    

def relu(x):
    return (x+np.abs(x))/2

def gen_threshold_ensemble(thresh=0.0):
    @two_argument
    @normalise("zscore")
    def threshold_ensemble(q,q0,*args,**kwargs):
        """See Aggarwal and Sathe (2015) <doi:10.1145/2830544.2830549> for more informations"""
        return np.sum(relu(q-thresh)+thresh,axis=-1)
    return threshold_ensemble

threshold_ensemble=gen_threshold_ensemble(0.0)
threshold_ensemble_1=gen_threshold_ensemble(1.0)
threshold_ensemble_2=gen_threshold_ensemble(2.0)
threshold_ensemble_minus_1=gen_threshold_ensemble(-1.0)
threshold_ensemble_minus_2=gen_threshold_ensemble(-2.0)

#sklearn.cluster.AffinityPropagation
from sklearn.cluster import AffinityPropagation

@two_argument
@normalisation_zscore
def clustered_weights(q,q0):
    """Computes an ensemble score using inverse cluster weighted averaging method by Chiang et al (2017)"""
    
    q=q.T
    q0=q0.T
    clustering = AffinityPropagation().fit(q)#calcon?
    assignments = clustering.predict(q)
    count=len(set(assignments))
    premean=[[i for i in range(len(q)) if assignments[i]==j] for j in range(count)]
    sb=[np.mean(q[premean[i]],axis=0) for i in range(count)]
    ret=np.mean(sb,axis=0)
    return ret

def wmean(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def wcov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - wmean(x, w)) * (y - wmean(y, w))) / np.sum(w)

def wcorr(x, y, w):
    """Weighted Correlation"""
    return wcov(x, y, w) / np.sqrt(wcov(x, x, w) * wcov(y, y, w))



def gen_greedy_ensemble(k=5):
    @two_argument
    @normalise("minmax")
    def _greedy_ensemble(q,q0):
        """See Schubert, Wojdanowski 2012"""

        n=len(q)
        q=q.T
        q0=q0.T
        #construct target
        indices=np.unique(np.concatenate([np.argsort(-qq)[:k] for qq in q]))
        target=np.zeros(n)
        target[indices]=1

        #compute weights
        weights=np.ones(n)/(2*(n-k))
        weights[indices]=1/(2*k)
        #for index in indices:
        #    target[index]=1
        #    weights[index]=1/(2*k)

        #compute correlation
        correlations= np.array([wcorr(qq,target,weights) for qq in q])

        cordex=np.argmax(correlations)
        ensemble=np.expand_dims(q[cordex],axis=0)
        qsub=np.delete(q,cordex,axis=0)
        optimal=correlations[cordex]
        
        while True:
            correlations= np.array([wcorr(qq,ensemble,weights) for qq in qsub])
            for i in np.argsort(correlations):#why do we not take abs(correlations)?
                test_ensemble=np.concatenate([ensemble,np.expand_dims(qsub[i],axis=0)],axis=0)
                test_scores=np.mean(test_ensemble,axis=0)
                new_correlation=wcorr(test_scores,target,weights)
                if new_correlation>optimal:
                    optimal=new_correlation
                    ensemble=test_ensemble
                    qsub=np.delete(qsub,i,axis=0)
                    break
            else:#we are done, if there did not occur any breaks before
                break
        return np.mean(ensemble,axis=0)


    return _greedy_ensemble

greedy_ensemble=gen_greedy_ensemble(5)

#g.multidimensional_grm_mml
#from girth import multidimensional_grm_mml as grm
from crm import crm
@two_argument
@normalise("min_max",usual_calcon=1)
def irt_ensemble(q,q0):
    """Computes an ensemble score using a continous response model"""
    q=q/(1+1e-6)
    q0=q0/(1+1e-6)
    thetas=crm(q)[0]
    return thetas



combinations={"maximum":maximum,
              "maximum_zscore":maximum_zscore,
              "maximum3":maximum3,
              "maximum4":maximum4,
              "maximum5":maximum5,
              "minimum":minimum,
              "minimum_zscore":minimum_zscore,
              "minimum3":minimum3,
              "minimum4":minimum4,
              "minimum5":minimum5,
              "mean":mean,
              "mean_zscore":mean_zscore,
              "mean3":mean3,
              "mean4":mean4,
              "mean5":mean5,
              "median1":median,
              "median2":median_beta,
              "median3":median_gamma,
              "median4":median_delta,
              "median5":median_epsilon,
              "l2mean":l2mean,
              "l2mean_zscore":l2mean_zscore,
              "l2mean3":l2mean3,
              "l2mean4":l2mean4,
              "l2mean5":l2mean5,
              "l3mean":l3mean,
              "l3mean_zscore":l3mean_zscore,
              "l3mean3":l3mean3,
              "l3mean4":l3mean4,
              "l3mean5":l3mean5,
              "l4mean":l4mean,
              "l4mean_zscore":l4mean_zscore,
              "l4mean3":l4mean3,
              "l4mean4":l4mean4,
              "l4mean5":l4mean5,
              "knn1":knn1,
              "knn3":knn3,
              "knn5":knn5,
              "knn10":knn10,
              "knn100":knn100,
              "gaussian_fit":gaussian_fit,
              "ifor_fit":ifor_fit,
              "threshold_ensemble":threshold_ensemble,
              "threshold_ensemble_1":threshold_ensemble_1,
              "threshold_ensemble_2":threshold_ensemble_2,
              "threshold_ensemble_minus_1":threshold_ensemble_minus_1,
              "threshold_ensemble_minus_2":threshold_ensemble_minus_2,
              "clustered_weights":clustered_weights,
              "greedy_ensemble":greedy_ensemble,
              "irt_ensemble":irt_ensemble,
              }


whats=list(combinations.keys())














