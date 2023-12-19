import torch
import torch.optim as optim 

X = torch.FloatTensor([[1], [2], [3]])
Y = torch.FloatTensor([[2], [4], [6]])  

W = torch.ones(1, requires_grad = True) 
b = torch.zeros(1, requires_grad = True)    


epochs = 1000
optimizer = optim.SGD([W, b], lr = 0.01)

for epoch in range(1, epochs + 1):
    hypothesis = X * W + b
    cost = torch.mean((hypothesis - Y) ** 2)
    
    optimizer.zero_grad()
    cost.backward() 
    optimizer.step()    
    
    if epoch % 100 == 0:
        print("Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}".format(
            epoch, epochs, W.item(), b.item(), cost.item()
        ))  
    
    