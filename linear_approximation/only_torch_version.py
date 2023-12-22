import torch



X = torch.FloatTensor([[1], [2], [3]])
Y = torch.FloatTensor([[2], [4], [6]])

fc1 = torch.nn.Linear(1, 1, bias=False)   
mse = torch.nn.MSELoss()
optimizer = torch.optim.SGD(fc1.parameters(), lr=0.1)   
epochs = 10
for epoch in range(epochs):
    mse_loss = mse(fc1(X), Y)
    optimizer.zero_grad()   
    mse_loss.backward()
    optimizer.step()
    
    print('Epoch {:4d}, W: {:.3f}, Cost: {:.6f}'.format(epoch, fc1.weight.item(), mse_loss.item())) 