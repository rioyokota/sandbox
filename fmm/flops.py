import numpy as np
N = 100000
M = 200
P = 10

tree = round(200*N*np.log(N))
P2M = 10*N*P*P
M2M = 30*(N/M)*P*P*P*P
M2L = 189*150*(N/M)*P*P*P
L2L = 30*(N/M)*P*P*P*P
L2P = 10*N*P*P
P2P = 20*27*N*M
total = tree + P2M + M2M + M2L + L2L + L2P + P2P
print(f'Tree : {tree:.1e} [FLOPs]')
print(f'P2M  : {P2M:.1e} [FLOPs]')
print(f'M2M  : {M2M:.1e} [FLOPs]')
print(f'M2L  : {M2L:.1e} [FLOPs]')
print(f'L2L  : {L2L:.1e} [FLOPs]')
print(f'L2P  : {L2P:.1e} [FLOPs]')
print(f'P2P  : {P2P:.1e} [FLOPs]')
print(f'Total: {total:.1e} [FLOPs]')
