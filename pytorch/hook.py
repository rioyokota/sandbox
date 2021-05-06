import torch

def printgradnorm(self, grad_input, grad_output):
    print('d loss / d y_p:', grad_output[0])
    print('d loss / d x  :', grad_input[0])
    print('d loss / d w  :', grad_input[1])

x = torch.tensor([[1.],[2.]],requires_grad=True)
y = torch.tensor([[1.],[1.]])
model = torch.nn.Linear(1,1,bias=False)
for param in model.parameters():
    torch.nn.init.constant_(param,2)
model.register_backward_hook(printgradnorm)
criterion = torch.nn.MSELoss(reduction='mean')
y_p = model(x)
loss = criterion(y_p, y)
print(loss)
loss.backward()
for param in model.parameters():
    print(param.grad)

