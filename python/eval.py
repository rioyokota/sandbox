import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

models = ['llmjp3-1.8b','llmjp3-3.7b','llmjp3-7.2b','llmjp3-13b','llmjp3-172b']
tasks = ['AVG','EL','FA','HE','MC','MR','MT','NLI','QA','RC','SUM']

config = pd.read_csv('models.csv',index_col=1)
scores = pd.read_csv('scores.csv',index_col=0)
ax = []
for task in tasks:
    plt.figure(figsize=(8,8),tight_layout=True)
    ax.append(plt.subplot(211))
    ax.append(plt.subplot(212))

for model in models:
    batch_size = config.batch_size[model]
    max_sequence_length = config.max_sequence_length[model]
    loss = pd.read_csv(model+'.csv',index_col=0)
    loss_steps = loss['steps'].to_numpy()
    loss_value = loss['loss'].to_numpy()
    score = scores.groupby(['model']).get_group((model,)).sort_values(by='iteration')
    eval_steps = score['iteration'].to_numpy()
    tokens = eval_steps * batch_size * max_sequence_length / 1e9
    loss_value = np.interp(eval_steps,loss_steps,loss_value)
    for i,task in enumerate(tasks):
        eval_value = score[task].to_numpy()
        ax[2*i+0].semilogx(tokens,eval_value)
        ax[2*i+1].plot(loss_value,eval_value,'o')

for i,task in enumerate(tasks):
    ax[2*i+0].set_xlim((1,3000))
    ax[2*i+0].grid(which='major')
    ax[2*i+0].grid(which='minor',linestyle='--')
    ax[2*i+0].set_xlabel('Trained tokens [$10^9$]')
    ax[2*i+0].set_ylabel('llm-jp-eval 1.4.1 - '+task)
    ax[2*i+0].legend(models)
    
    ax[2*i+1].set_xlim((1,10))
    ax[2*i+1].grid(which='major')
    ax[2*i+1].grid(which='minor',linestyle='--')
    ax[2*i+1].set_xlabel('Train loss')
    ax[2*i+1].set_ylabel('llm-jp-eval 1.4.1 - '+task)
    ax[2*i+1].invert_xaxis()
plt.show()
