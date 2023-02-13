
import sys

import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam

import numpy as np
import os
import shutil


class DEAN():
    def __init__(
            
        self,
        lr:float= 0.03,
        batch:int= 100,
        depth:int= 3,
        bag:int= 10,
        index:int=0,
        rounds:int= 100,

    ):
        
        self.lr = lr
        self.batch = batch
        self.depth = depth
        self.bag = bag
        self.index = index
        self.rounds = rounds



    #train one model (of index dex)
    def train(self, dex, x,  tx, y, ty):
        
        pth=f"results/{dex}/"
        
        
        if os.path.isdir(pth):
            shutil.rmtree(pth)
        
        os.makedirs(pth, exist_ok=False)
        
        
        def statinf(q):
            return {"shape":q.shape,"mean":np.mean(q),"std":np.std(q),"min":np.min(q),"max":np.max(q)}
        
        
        

        x_train, x_test = x.copy(), tx.copy()


        #choose some features for the current model
        predim=int(x_train.shape[1])

        if self.bag > predim:
            self.bag = predim

        to_use=np.random.choice([i for i in range(predim)],self.bag,replace=False)

        x_train=np.concatenate([np.expand_dims(x_train[:,use],axis=1) for use in to_use],axis=1)
        x_test =np.concatenate([np.expand_dims(x_test [:,use],axis=1) for use in to_use],axis=1)


        #normalise the data, so that the mean is zero, and the standart deviation is one
        norm=np.mean(x_train)
        norm2=np.std(x_train)

        def normalise(q):
            return (q-norm)/norm2
        

        def getdata(x,y,norm=True,normdex=7,n=-1):
            if norm:
                ids=np.where(y==normdex)
            else:
                ids=np.where(y!=normdex)
            qx=x[ids]
            if n>0:qx=qx[:n]
            qy=np.reshape(qx,(int(qx.shape[0]),self.bag))
            return normalise(qy)
        
        #split data into normal and abnormal samples. Train only on normal ones
        normdex= self.index
        train=getdata(x_train,y,norm=True,normdex=normdex)
        at=getdata(x_test,ty,norm=False,normdex=normdex)
        t=getdata(x_test,ty,norm=True,normdex=normdex)
        
    
        #function to build one tensorflow model 
        def getmodel(q,reg=None,act="relu",mean=1.0):
            inn=Input(shape=(self.bag,))
            w=inn
            for aq in q[1:-1]:
                #change this line to use constant shifts
                #w=Dense(aq,activation=act,use_bias=True,kernel_initializer=keras.initializers.TruncatedNormal(),kernel_regularizer=reg)(w)
                w=Dense(aq,activation=act,use_bias=False,kernel_initializer=keras.initializers.TruncatedNormal(),kernel_regularizer=reg)(w)
            w=Dense(q[-1],activation="linear",use_bias=False,kernel_initializer=keras.initializers.TruncatedNormal(),kernel_regularizer=reg)(w)
            m=Model(inn,w,name="oneoff")
            zero=K.ones_like(w)*mean
            loss=mse(w,zero)
            loss=K.mean(loss)
            m.add_loss(loss)
            m.compile(Adam(learning_rate= self.lr))
            return m
        
        l=[self.bag for i in range(self.depth)]
        m=getmodel(l,reg=None,act="relu",mean=1.0)
        
        cb=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True),
                        keras.callbacks.TerminateOnNaN()]
        cb.append(keras.callbacks.ModelCheckpoint(f"{pth}/model.tf", monitor='val_loss', verbose=0, save_best_only=True,save_weights_only=True))
        
        
        #train the model    
        h=m.fit(train,None,
                epochs=500,
                batch_size= self.batch,
                validation_split=0.25,
                verbose=0,
                callbacks=cb)

        #predict the output of our datasets 
        pain=m.predict(train)
        p=m.predict(t)
        w=m.predict(at)
    
        #average out the last dimension, to get one value for each samples 
        ppain=np.mean(pain,axis=-1)
        pp=np.mean(p,axis=-1)
        ww=np.mean(w,axis=-1)
    
        #calculate the mean prediction (q in the paper) 
        m=np.mean(ppain)

        #and the deviation of each to the mean
        pd=np.abs(pp-m)#if this worked, the values in the array pd should be much smaller
        wd=np.abs(ww-m)#than in the array wd
        y_score=np.concatenate((pd,wd))
        y_true=np.concatenate((np.zeros_like(pp),np.ones_like(ww)))
        
        #calculate auc score of a single model
        #auc_score=auc(y_true,y_score)
        #print(f"reached auc of {auc_score}")
        
        #and save the necessary results for merge.py to combine the submodel predictions into an ensemble
        #np.savez_compressed(f"{pth}/result.npz",y_true=y_true,y_score=y_score,to_use=to_use)
        
        return y_true, y_score
    


    def fit(self, xtrain, ytrain, xtest, ytest):
        y_scores = []
        pwr = 0

        for dex in range(0,self.rounds):
            y_true, y_scor = self.train(dex, x = xtrain, y = ytrain, tx = xtest, ty = ytest)
            y_scores.append(y_scor)


        wids=[np.std(y_score[np.where(y_true==0)]) for y_score in y_scores]

        y_score=np.sqrt(np.mean([(y_score/wid**pwr)**2 for y_score,wid in zip(y_scores,wids)],axis=0))

        return y_true,y_score