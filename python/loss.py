import pandas as pd
import matplotlib.pyplot as plt
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
for i,model in enumerate(models):
    batch_size = config.batch_size[model]
    max_sequence_length = config.max_sequence_length[model]
    runs = api.runs(teams[i]+projects[i],filters=filters[i])
    print(f'From project {teams[i]+projects[i]}')
    plt.figure(i)
    ax = plt.subplot()
    steps = []
    tokens = []
    loss = []

    for run in runs:
        history = run.history(samples=samples)
        samples = 1000
        if '_step' in history:
            print(f'Run {run.name}: Steps {history._step.min()} to {history._step.max()}')
            history['tokens'] = history['_step'] * batch_size * max_sequence_length / 1e9
            history.plot(ax=ax,x='tokens',y='lm-loss-training/lm loss',logx=True,logy=True)
            steps = steps + history['_step'].to_list()
            tokens = tokens + history['tokens'].to_list()
            loss = loss + history['lm-loss-training/lm loss'].to_list()
        else:
            print(f'Run {run.name} does not have any logs.')

    ax.set_xlim((1,3000))
    ax.set_ylim((1,10))
    ax.grid(which='major')
    ax.grid(which='minor',ls='--')
    ax.set_xlabel('Trained tokens [$10^9$]')
    ax.set_ylabel('Train loss')
    ax.get_legend().remove()
    ax.set_title(model)

    plt.figure(5)
    ax = plt.subplot()
    ax.loglog(tokens,loss,label=model)
    ax.set_xlim((1,3000))
    ax.set_ylim((1,10))
    ax.grid(which='major')
    ax.grid(which='minor',ls='--')
    ax.set_xlabel('Trained tokens [$10^9$]')
    ax.set_ylabel('Train loss')
    ax.legend()

    data = {'steps':steps,'loss':loss}
    df = pd.DataFrame(data)
    df.to_csv(model+'.csv')
plt.show()
