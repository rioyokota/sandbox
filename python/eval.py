import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

models = ['llmjp3-1.8b','llmjp3-3.7b','llmjp3-7.2b','llmjp3-13b','llmjp3-172b']
tasks = ['AVG','EL','FA','HE','MC','MR','MT','NLI','QA','RC','SUM']

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
        ax[2*i+1].plot(loss_value,eval_value,'.')

eval_min = [0.03,0,0,0,0.2,0,0.33,0.33,0,0,0]
eval_max = [0.6,0.7,0.31,0.7,0.85,0.9,0.83,0.7,0.7,0.88,0.11]
gpt4 = [0.624,0.631,0.360,0.755,0.923,0.940,0.836,0.788,0.613,0.893,0.126]
gpt3 = [0.537,0.588,0.258,0.575,0.780,0.860,0.828,0.554,0.486,0.871,0.104]
models.extend(['gpt-3.5','gpt-4'])
x_loss = np.arange(1, 4.1, 0.1)
x_tokens = np.arange(1, 3000)

def sigmoid(x, x0, k):
    return (1 / (1 + np.exp(-k*(x0-x))))

for i,task in enumerate(tasks):
    ax[2*i+0].semilogx(x_tokens,[gpt3[i]]*len(x_tokens),'--')
    ax[2*i+0].semilogx(x_tokens,[gpt4[i]]*len(x_tokens),'--')
    ax[2*i+0].set_xlim((1,3000))
    ax[2*i+0].grid(which='major')
    ax[2*i+0].grid(which='minor',ls='--')
    ax[2*i+0].set_xlabel('Trained tokens [$10^9$]')
    ax[2*i+0].set_ylabel('llm-jp-eval 1.4.1 - '+task)
    ax[2*i+0].legend(models)
    
    x_train = np.array(all_loss)
    y_train = np.array(all_eval[task])
    x_range = (1.7 < x_train) & (x_train < 8)
    y_train = y_train[x_range]
    x_train = x_train[x_range]
    y_train = (y_train - eval_min[i])/(eval_max[i]-eval_min[i])
    popt,pcov = curve_fit(sigmoid,x_train,y_train)
    y_eval = sigmoid(x_loss,*popt)*(eval_max[i]-eval_min[i])+eval_min[i]
    if i == 0:
        y_eval_avg = np.zeros(len(x_loss))
    else:
        y_eval_avg = y_eval_avg + y_eval
    ax[2*i+1].plot(x_loss,[gpt3[i]]*len(x_loss),'--')
    ax[2*i+1].plot(x_loss,[gpt4[i]]*len(x_loss),'--')
    ax[2*i+1].plot(x_loss,y_eval,'-')
    ax[2*i+1].set_xlim((1,4))
    ax[2*i+1].grid(which='major')
    ax[2*i+1].grid(which='minor',ls='--')
    ax[2*i+1].set_xlabel('Train loss')
    ax[2*i+1].set_ylabel('llm-jp-eval 1.4.1 - '+task)
    ax[2*i+1].legend(models+['Fit of '+task])
    ax[2*i+1].invert_xaxis()

y_eval_avg = y_eval_avg / 11
ax[1].plot(x_loss,y_eval_avg,'-')
ax[1].legend(models+['Fit of AVG','AVG of Fits'])
data = {'loss':x_loss,'eval':y_eval_avg}
df = pd.DataFrame(data)
df.to_csv('loss2eval.csv')
plt.show()
