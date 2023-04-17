import numpy as np


basics={"mean":np.mean,
        "median":np.median,
        "min":np.min,
        "max":np.max,
        "len":len,
        "std":np.std,
        "frac":lambda x: len([zw for zw in x if zw>0.5])/len(x)}

thans=["mean","median","min","max"]


