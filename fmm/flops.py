import numpy as np
N = 100000 # Number of particles
M = 10 # Number of particles per leaf
P = 10 # Order of expansion
n = 3
P2Pneighbor = n * n * n
M2Lneighbor = 6 * 6 * 6 - P2Pneighbor

tree = round(200*N*np.log(N))
P2M = 10*N*P*P
M2M = 30*(N/M)*P*P*P*P
M2L = M2Lneighbor*150*(N/M)*P*P*P
L2L = 30*(N/M)*P*P*P*P
L2P = 10*N*P*P
P2P = 20*P2Pneighbor*N*M
total = tree + P2M + M2M + M2L + L2L + L2P + P2P
print(f'Tree : {tree:.1e} [FLOPs]')
print(f'P2M  : {P2M:.1e} [FLOPs]')
print(f'M2M  : {M2M:.1e} [FLOPs]')
print(f'M2L  : {M2L:.1e} [FLOPs]')
print(f'L2L  : {L2L:.1e} [FLOPs]')
print(f'L2P  : {L2P:.1e} [FLOPs]')
print(f'P2P  : {P2P:.1e} [FLOPs]')
print(f'Total: {total:.1e} [FLOPs]')
