import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
G = []
plt.figure()
ax = plt.subplot()
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
    loss_value = np.interp(eval_steps,loss_steps,loss_value).tolist()
    N.extend([model_size]*len(tokens))
    D.extend(tokens)
    L.extend(loss_value)
    G.extend(gpu_months)
    ax.loglog(gpu_months,loss_value)
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
#A = 406.4
#B = 410.7
#E = 1.69
#alpha = 0.34
#beta = 0.28
Law = E + A/np.array(N)**alpha + B/np.array(D)**beta
ax.loglog(G,Law,'o')
print(f'A: {A}, B: {B}, E:{E}, alpha: {alpha}, beta: {beta}')

plt.show()
