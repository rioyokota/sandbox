import pandas as pd
import matplotlib.pyplot as plt

scores = pd.read_csv('scores.csv',index_col=0)
config = pd.read_csv('models.csv',index_col=1)
models = ['llmjp3-1.8b','llmjp3-3.7b','llmjp3-7.2b','llmjp3-13b','llmjp3-172b']

plt.figure()
ax = plt.subplot()
print(scores.keys())
for model in models:
    batch_size = config.batch_size[model]
    max_sequence_length = config.max_sequence_length[model]
    score = scores.groupby(['model']).get_group((model,)).sort_values(by='iteration')
    score['tokens'] = score.iteration * batch_size * max_sequence_length / 1e9
    score.plot(ax=ax,x='tokens',y='AVG',logx=True,label=model)
ax.set_xlim((1,3000))
ax.grid(which='major')
ax.grid(which='minor',linestyle='--')
ax.set_xlabel('Trained tokens [$10^9$]')
ax.set_ylabel('llm-jp-eval 1.4.1 - AVG')
plt.show()
