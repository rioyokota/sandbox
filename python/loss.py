import wandb

api = wandb.Api()
runs = api.runs("nii-geniac/Llama-2-175B",filters={"tags":"main"})
print(runs[0].history().keys())
for run in runs:
    history = run.history()
    if '_step' in history:
        print(history._step.min(),history._step.max(),history['lm-loss-training/lm loss'].max(),history['lm-loss-training/lm loss'].min())
    else:
        print(run.name)
