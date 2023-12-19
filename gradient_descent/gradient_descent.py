import torch

X = torch.FloatTensor([[1], [2], [3]])
Y = torch.FloatTensor([[2], [4], [6]])

W = torch.zeros(1)    
lr = 0.1

epochs = 5
for epoch in range(1, epochs + 1):
    hypothesis = W * X
    cost = torch.mean((hypothesis - Y) ** 2)
    gradient = torch.sum(2 * torch.mean((W * X - Y) * X))
    
    print("Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}".format(epoch, epochs, W.item(), cost.item()))    
    
    W -= lr * gradient  
    
    