import numpy as np

def whattrafo(what,dic):
    return dic["merge"][what]
def thantrafo(than,dic):
    return dic["stats"][than]



def is_bigger_than(what, than):
    def f(dic):
        return whattrafo(what,dic) >= thantrafo(than,dic)
    return f

def average_distance(what, than):
    def f(dic):
        return whattrafo(what,dic) - thantrafo(than,dic)
    return f

def auc_distance(what,than):
    def f(dic):
        auc1= whattrafo(what,dic)
        auc2= thantrafo(than,dic)
        return np.log(1-auc2+1e-10) - np.log(1-auc1+1e-10)
        #return 2*(auc1-auc2)/abs((2-auc1-auc2)+1e-6)
    return f

def direct_auc(what):
    def f(dic):
        return whattrafo(what,dic)
    return f



def all_funcs(whats,thens):
    dic={}
    for what in whats:
        dic[f"directAuc_"+what]=direct_auc(what)
        for than in thens:
            dic["bigger_" +what+"_"+than]=is_bigger_than(what,than)
            dic["average_" + what+"_"+than]=average_distance(what,than)
            dic["aucdistance_"+what+"_"+than]=auc_distance(what,than)
    return dic



