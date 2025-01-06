import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def sigmoid(x, x0, k):
    return (1 / (1 + np.exp(-k*(x0-x))))

models = ['llmjp3-1.8b','llmjp3-3.7b','llmjp3-7.2b','llmjp3-13b','llmjp3-172b']
tasks = ['AVG','EL','FA','HE','MC','MR','MT','NLI','QA','RC','SUM']
#tasks = ['AVG']

config = pd.read_csv('models.csv',index_col=1)
scores = pd.read_csv('scores.csv',index_col=0)
ax = []
all_eval = {}
for task in tasks:
    plt.figure(figsize=(8,8),tight_layout=True)
    ax.append(plt.subplot(211))
    ax.append(plt.subplot(212))
    all_eval.update({task:[]})

all_loss = []
for model in models:
    batch_size = config.batch_size[model]
    max_sequence_length = config.max_sequence_length[model]
    loss = pd.read_csv(model+'.csv',index_col=0)
    loss_steps = loss['steps'].to_numpy()
    loss_value = loss['loss'].to_numpy()
    score = scores.groupby(['model']).get_group((model,)).sort_values(by='iteration')
    eval_steps = score['iteration'].to_numpy()
    tokens = eval_steps * batch_size * max_sequence_length / 1e9
    loss_value = np.interp(eval_steps,loss_steps,loss_value).tolist()
    all_loss.extend(loss_value)
    for i,task in enumerate(tasks):
        eval_value = score[task]
        all_eval[task].extend(eval_value)
        ax[2*i+0].semilogx(tokens,eval_value)
        ax[2*i+1].plot(loss_value,eval_value,'o')

eval_min = [0.03,0,0,0,0.2,0,0.33,0.33,0,0,0]
eval_max = [0.6,0.7,0.4,0.7,0.85,0.9,0.83,0.7,0.7,0.88,0.11]
x = np.arange(1, 10, 0.1)

for i,task in enumerate(tasks):
    ax[2*i+0].set_xlim((1,3000))
    ax[2*i+0].grid(which='major')
    ax[2*i+0].grid(which='minor',linestyle='--')
    ax[2*i+0].set_xlabel('Trained tokens [$10^9$]')
    ax[2*i+0].set_ylabel('llm-jp-eval 1.4.1 - '+task)
    ax[2*i+0].legend(models)
    
    x_train = np.array(all_loss[0:2750])
    y_train = np.array(all_eval[task][0:2750])
    y_train = y_train[x_train < 4]
    x_train = x_train[x_train < 4]
    y_train = (y_train - eval_min[i])/(eval_max[i]-eval_min[i])
    popt,pcov = curve_fit(sigmoid,x_train,y_train)
    if task == 'EL':
        popt[1] = 5
    y = sigmoid(x,*popt)*(eval_max[i]-eval_min[i])+eval_min[i]
    ax[2*i+1].plot(x,y,'-')
    ax[2*i+1].set_xlim((1,4))
    ax[2*i+1].grid(which='major')
    ax[2*i+1].grid(which='minor',linestyle='--')
    ax[2*i+1].set_xlabel('Train loss')
    ax[2*i+1].set_ylabel('llm-jp-eval 1.4.1 - '+task)
    ax[2*i+1].invert_xaxis()
plt.show()
