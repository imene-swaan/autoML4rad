import numpy as np

def loaddata():
    f=np.load("/global/cardio.npz")
    x = f['x']
    a = (x.shape[0],) 
    y = np.zeros(a)

    return (f["x"], y),(f["tx"],f["ty"])

if __name__=="__main__":
    x,(tx,ty)=loaddata()

    a = (x.shape[0], ) 
    y = np.zeros(a)

    print(y.shape)
