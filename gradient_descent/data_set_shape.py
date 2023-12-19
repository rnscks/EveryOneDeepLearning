import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch

X = torch.FloatTensor([[1], [2], [3]])
Y = torch.FloatTensor([[1], [2], [3]])

print("X shape")
print(X.shape)


print("Y shape")
print("Y shape")


from matplotlib import pyplot as plt
import numpy as np

plt.scatter(X[:, 0].numpy(), Y[:, 0].numpy())   
plt.scatter(X[:, 0].numpy(), np.zeros_like(X[:, 0].numpy()))
plt.scatter(np.zeros_like(Y[:, 0].numpy()), Y[:, 0].numpy())
plt.show()
