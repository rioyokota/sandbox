import torch
x = torch.tensor([[1.],[2.]],requires_grad=True)
y = torch.tensor([[0.],[0.]])
model = torch.nn.Linear(1,1,bias=False)
for param in model.parameters():
    torch.nn.init.constant_(param,1)
criterion = torch.nn.MSELoss(reduction='mean')
y_p = model(x)
loss = criterion(y_p, y)
print(loss)
loss.backward()
for param in model.parameters():
    print(param.grad)
