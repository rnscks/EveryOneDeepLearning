import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


from matplotlib import pyplot as plt    
import torch


X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = torch.FloatTensor([[0], [1], [1], [0]])


plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c = Y[:, 0].numpy())
plt.show()