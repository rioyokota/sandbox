import matplotlib.pyplot as plt
import numpy as np

# Billion parameters
params = np.array([1.3, 7, 13, 70, 172, 70*8, 172*8])
isoFLOPs_levels = [5e23, 1e24, 2e24]
flops_per_param_token = 550 * 10**12 * 2500 * 30 * 24 * 2600 / (172 * 2000)

def calculate_isoFLOPs(params,tokens):
    N = params * 10**9
    D = tokens * 10**9
    E = 1.69
    A = 406.4
    B = 410.7
    alpha = 0.34
    beta = 0.28
    return E + A / N**alpha + B / D**beta

plt.figure(figsize=(10, 6))

colors = ['r', 'g', 'b', 'c', 'm']
for i,flops in enumerate(isoFLOPs_levels):
    tokens = flops / flops_per_param_token / params
    iso_loss = calculate_isoFLOPs(params,tokens)
    print(iso_loss)
    plt.plot(params, iso_loss, 'o-', color=colors[i], label=f'IsoFLOPs {flops:.0e}')

plt.xscale('log')
plt.xlabel('Number of Parameters (in billions)')
plt.ylabel('Training Loss')
plt.title('IsoFlops Curves: Training Loss vs Number of Parameters')
plt.legend()
plt.grid(True)
plt.show()

