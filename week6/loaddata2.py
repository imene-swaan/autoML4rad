import numpy as np

def loaddata(dataset_name):
    f=np.load('/global/datasets/' +  dataset_name + '.npz')
    x = f['x']
    a = (x.shape[0],) 
    y = np.zeros(a)

    return (f["x"], y),(f["tx"],f["ty"])

if __name__=="__main__":
    (x,y),(tx,ty)=loaddata()

    print(x.shape)
