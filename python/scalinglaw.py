import matplotlib.pyplot as plt
import numpy as np

# Billion parameters
params = np.array([1.3, 7, 13, 70, 172, 460, 1110])
gpu_months = np.array([100, 1000, 10000, 100000])
isoFLOPs_levels = gpu_months * 430. * 10**12 * 3600 * 24 * 30
flops_per_param_token = 430 * 10**12 * 3600 * 24 * 30 * 2600 / (172 * 2000)

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
    plt.plot(params, iso_loss, 'o-', color=colors[i], label=f'{gpu_months[i]} GPU months')

plt.xscale('log')
plt.xlabel('Number of Parameters (in billions)')
plt.ylabel('Training Loss')
plt.title('IsoFlops Curves: Training Loss vs Number of Parameters')
plt.legend()
plt.grid(True)
plt.show()

