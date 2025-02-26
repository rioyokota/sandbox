d = 1 # data parallel size
p = 1 # pipeline parallel size
t = 1 # tensor parallel size
c = 1 # context parallel size
v = 99487 # vocabulary size
e = 8 # number of experts
m = 13 # model size

if m == 13:
    a = 40 # number of attention heads
    h = 5120 # hidden size
    f = 13824 # FFN intermediate size
    l = 40 # number of hidden layers
    s = 4096 # sequence length
    k = 40 # number of key-value heads
elif m == 70:
    a = 64 # number of attention heads
    h = 8192 # hidden size
    f = 28672 # FFN intermediate size
    l = 80 # number of hidden layers
    s = 4096 # sequence length
    k = 8 # number of key-value heads
elif m == 172:
    a = 96 # number of attention heads
    h = 12288 # hidden size
    f = 38464 # FFN intermediate size
    l = 96 # number of hidden layers
    s = 4096 # sequence length
    k = 16 # number of key-value heads

parameters = h*v/t+2*l/p*h**2*((1+k/a+e*3./2*f/h)/t+1/h)
print(parameters/1e9)
