import numpy as np



def modulo_name(i, lis):
    if len(lis)==0:
        return str(i)
    return modulo_name(i // lis[0], lis[1:]) +"/"+  str(i % lis[0])


if __name__=="__main__":
    lis=[10, 10, 10, 10]
    for i in range(0,10000,1):
        print(modulo_name(i, lis))



