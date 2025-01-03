import pandas as pd
import matplotlib.pyplot as plt
import wandb

api = wandb.Api()
config = pd.read_csv('models.csv',index_col=1)
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

for i in range(1):
    runs = api.runs(teams[i]+projects[i],filters=filters[i])
    print(f'From project {teams[i]+projects[i]}')

    plt.figure()
    ax = plt.subplot()
    for it,run in enumerate(runs):
        batch_size = config.batch_size[models[i]]
        max_sequence_length = config.max_sequence_length[models[i]]
        run_history = run.history()
        if '_step' in run_history:
            print(f'Run {run.name}: Steps {run_history._step.min()} to {run_history._step.max()}')
            run_history['tokens'] = run_history._step * batch_size * max_sequence_length / 1e9
            run_history.plot(ax=ax,x='tokens',y='lm-loss-training/lm loss',logx=True,logy=True)
        else:
            print(f'Run {run.name} does not have any logs.')
    ax.set_xlim((1,3000))
    ax.set_ylim((1,10))
    ax.grid(which='major')
    ax.grid(which='minor',linestyle='--')
    ax.set_xlabel('Trained tokens [$10^9$]')
    ax.set_ylabel('Train loss')
    ax.get_legend().remove()
    ax.set_title(models[i])
plt.show()
