import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from itertools import compress
from scipy.optimize import minimize
from scipy.special import huber

models = ['llmjp3-1.8b','llmjp3-3.7b','llmjp3-7.2b','llmjp3-13b','llmjp3-172b']
h100_flop_per_month = 430 * 10**12 * 3600 * 24 * 30
flops_per_param_token = h100_flop_per_month * 2600 / (172e9 * 2e12) 

config = pd.read_csv('models.csv',index_col=1)
scores = pd.read_csv('scores.csv',index_col=0)
N = []
D = []
L = []
plt.figure(figsize=(8,8),tight_layout=True)
ax = plt.subplot(211)
for model in models:
    batch_size = config.batch_size[model]
    max_sequence_length = config.max_sequence_length[model]
    model_size = config.approx_model_size[model]
    loss = pd.read_csv(model+'.csv',index_col=0)
    loss_steps = loss['steps'].to_numpy()
    loss_value = loss['loss'].to_numpy()
    score = scores.groupby(['model']).get_group((model,)).sort_values(by='iteration')
    eval_steps = score['iteration'].to_numpy()
    tokens = eval_steps * batch_size * max_sequence_length
    flops = flops_per_param_token * model_size * tokens
    gpu_months = flops / h100_flop_per_month
    loss_value = np.interp(eval_steps,loss_steps,loss_value)
    model_size = [model_size]*len(tokens)
    loss_range = (1.7 < loss_value) & (loss_value < 8)
    N.extend(compress(model_size,loss_range))
    D.extend(compress(tokens,loss_range))
    L.extend(compress(loss_value,loss_range))
    ax.loglog(gpu_months,loss_value,'o')
ax.set_ylim((1,20))
ax.grid(which='major')
ax.grid(which='minor',linestyle='--')
ax.set_xlabel('Training budget [GPU months]')
ax.set_ylabel('Train loss')
ax.legend(models)

def objective(params, N, D, L, delta=1e-3):
    a, b, e, alpha, beta = params
    term1 = a - alpha * np.log(N)
    term2 = b - beta * np.log(D)
    predicted_log_L = np.log(np.exp(term1) + np.exp(term2) + np.exp(e))
    true_log_L = np.log(L)
    residuals = true_log_L - predicted_log_L
    return np.sum(huber(delta, residuals))

initial_guess = [np.log(406.4), np.log(410.7), np.log(1.69), 0.34, 0.28]
result = minimize(objective,initial_guess,args=(N, D, L),method='L-BFGS-B')
a, b, e, alpha, beta = result.x
A = np.exp(a)
B = np.exp(b)
E = np.exp(e)

colors = list(mcolors.TABLEAU_COLORS)
for i,model in enumerate(models):
    batch_size = config.batch_size[model]
    max_sequence_length = config.max_sequence_length[model]
    model_size = config.approx_model_size[model]
    eval_steps = score['iteration'].to_numpy()
    tokens = eval_steps * batch_size * max_sequence_length
    flops = flops_per_param_token * model_size * tokens
    gpu_months = flops / h100_flop_per_month
    N = [model_size]*len(tokens)
    D = tokens
    L = E + A/np.array(N)**alpha + B/np.array(D)**beta
    ax.loglog(gpu_months,L,color=colors[i],ls='-')
print(f'A: {A}, B: {B}, E:{E}, alpha: {alpha}, beta: {beta}')

params = np.array([1.3, 7, 13, 70, 172, 460, 1110])
gpu_months = np.array([100, 1000, 10000, 100000])
isoFLOPs_levels = gpu_months * 430. * 10**12 * 3600 * 24 * 30
flops_per_param_token = 430 * 10**12 * 3600 * 24 * 30 * 2600 / (172 * 2000)

def calculate_isoFLOPs(params,tokens):
    N = params * 10**9
    D = tokens * 10**9
    return E + A / N**alpha + B / D**beta

ax = plt.subplot(212)
for i,flops in enumerate(isoFLOPs_levels):
    tokens = flops / flops_per_param_token / params
    iso_loss = calculate_isoFLOPs(params,tokens)
    ax.semilogx(params, iso_loss, 'o-', label=f'{gpu_months[i]} GPU months')

ax.set_xlabel('Number of Parameters (in billions)')
ax.set_ylabel('Training Loss')
ax.legend()
ax.grid(which='major')
ax.grid(which='minor',linestyle='--')
plt.show()
