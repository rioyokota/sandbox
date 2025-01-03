import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wandb

models = ['llmjp3-1.8b','llmjp3-3.7b','llmjp3-7.2b','llmjp3-13b','llmjp3-172b']
projects = ['nii-geniac-1.7B','v3-3.8b','v3-7.3b','Llama-2-13B','Llama-2-175B']
teams = ['llm-jp/','llm-jp/','llm-jp/','nii-geniac/','nii-geniac/']
filters = [
        {'tags':'exp2-main'},
        {'created_at':{'$lt':'2024-09-10'}},
        {},
        {'display_name':{'$regex':'(llama-2-13b-exp4-sakura-2024.*)|(llama-2-13b-exp4-2024.*)'}},
        {'tags':'main'}
        ]
samples = 10000

api = wandb.Api()
config = pd.read_csv('models.csv',index_col=1)
scores = pd.read_csv('scores.csv',index_col=0)
plt.figure()
ax = plt.subplot()
for i,model in enumerate(models):
    batch_size = config.batch_size[model]
    max_sequence_length = config.max_sequence_length[model]
    loss = pd.read_csv(model+'.csv',index_col=0)
    loss_steps = loss['steps'].to_numpy()
    loss_value = loss['loss'].to_numpy()
    score = scores.groupby(['model']).get_group((model,)).sort_values(by='iteration')
    eval_steps = score['iteration'].to_numpy()
    eval_value = score['AVG'].to_numpy()
    loss_value = np.interp(eval_steps,loss_steps,loss_value)
    ax.plot(loss_value,eval_value,'o')

ax.set_xlim((1,10))
ax.set_ylim((0,1))
ax.grid(which='major')
ax.grid(which='minor',linestyle='--')
ax.set_xlabel('Train loss')
ax.set_ylabel('llm-jp-eval 1.4.1 - AVG')
ax.set_title('AVG')
ax.invert_xaxis()
plt.show()
