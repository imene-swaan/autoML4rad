import numpy as np


from sklearn import tree

import matplotlib.pyplot as plt


#here we have two inputs, the number of features (dims) and the number of training samples (counts)
#and want to predict two outputs for trained knns
#the method (categorical)
#and k (numerical)

#data loads/imports all our previous training
from data import dims, counts, hyperparameters

inp=np.stack([dims,counts],axis=1)


n_estimators = hyperparameters['n_estimators']
max_samples = hyperparameters['max_samples']

#train a classifier to find the best n_estimators for a given input
clf1=tree.DecisionTreeClassifier(max_depth=2)
clf1=clf1.fit(inp,n_estimators)

#train a classifier to find the best max_samples for a given input
clf3=tree.DecisionTreeRegressor(max_depth=3)
clf3=clf3.fit(inp,max_samples)



#plot both as trees
plt.figure(figsize=(16,10))

plt.subplot(1,2,1)
tree.plot_tree(clf1,feature_names=['dims','counts'],filled=True)

plt.subplot(1,2,2)
tree.plot_tree(clf3,feature_names=['dims','counts'],filled=True)

plt.savefig('../results/week4/images/tree.png')
plt.savefig('../results/week4/images/tree.pdf')

plt.show(block=True)






inp1,inp2=np.exp(np.linspace(0,np.log(100),20)),np.exp(np.linspace(0,np.log(1000),20))
oinp1,oinp2=inp1,inp2

inp1,inp2=np.meshgrid(inp1,inp2)
inp=np.stack([inp1.flatten(),inp2.flatten()],axis=1)


from test_data import dims_test, counts_test
inp_test=np.stack([dims_test,counts_test],axis=1)




#plot the regions in which a number of estimators is best
outp=clf1.predict(inp)

outp=outp.reshape(inp1.shape)
plt.figure(figsize=(10,10))


plt.imshow(outp,origin='lower',cmap='viridis')
plt.xticks(np.linspace(0,19,5),np.round(np.exp(np.linspace(0,np.log(100),5)),1))
plt.yticks(np.linspace(0,19,5),np.round(np.exp(np.linspace(0,np.log(1000),5)),1))
plt.xlabel('dims')
plt.ylabel('counts')
cbar=plt.colorbar()
plt.savefig('../results/week4/images/estimators.png')
plt.savefig('../results/week4/images/estimators.pdf')
plt.show(block=True)





#plot the regions in which a number of samples is best
outp=clf3.predict(inp)
outp=outp.reshape(inp1.shape)
plt.figure(figsize=(10,10))

plt.imshow(outp,origin='lower',cmap='viridis')
plt.xticks(np.linspace(0,19,5),np.round(np.exp(np.linspace(0,np.log(100),5)),1))
plt.yticks(np.linspace(0,19,5),np.round(np.exp(np.linspace(0,np.log(1000),5)),1))
plt.xlabel('dims')
plt.ylabel('counts')
cbar=plt.colorbar()
plt.savefig('../results/week4/images/samples.png')
plt.savefig('../results/week4/images/samples.pdf')
plt.show(block=True)

print('done')

