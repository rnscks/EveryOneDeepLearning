import torch


W = torch.zeros(1, requires_grad = True)
b = torch.zeros(1, requires_grad = True)    

print("W shape")
print(W.shape)
print("W:" , W)

print("b shape")
print(b.shape)
print("b:" , b)