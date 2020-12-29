import torch

def printnorm(self, input, output):
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output data:', output.data)

x = torch.tensor([[1.],[2.]],requires_grad=True)
y = torch.tensor([[0.],[0.]])
model = torch.nn.Linear(1,1,bias=False)
for param in model.parameters():
    torch.nn.init.constant_(param,1)
model.register_forward_hook(printnorm)
criterion = torch.nn.MSELoss(reduction='mean')
y_p = model(x)
loss = criterion(y_p, y)
print(loss)
loss.backward()
for param in model.parameters():
    print(param.grad)

