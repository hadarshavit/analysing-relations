import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools

print(os.listdir())

def open_csv(path):
    df = pd.read_csv(path, names=['file', 'predicted', 'truth'])
    print(df.info())
    return df

def analyze_models(modela, modelb, modelc):
    at_least_one_correct = 0
    none = 0

    # for i in range(len(modela)):
    #     assert modela['file'][i] == modelb['file'][i]
    #     if modela['predicted'][i] == modela['truth'][i] or modelb['predicted'][i] == modelb['truth'][i] or modelc['predicted'][i] == modelc['truth'][i]:
    #         at_least_one_correct += 1
    #     else:
    #         none += 1
    assert (modela['file'] == modelb['file']).to_numpy().all()
    assert (modela['file'] == modelc['file']).to_numpy().all()

    modela_preds = (modela['predicted'] == modela['truth']).to_numpy()
    modelb_preds = (modelb['predicted'] == modelb['truth']).to_numpy()
    modelc_preds = (modelc['predicted'] == modelc['truth']).to_numpy()

    anything_correct = np.sum(modela_preds | modelb_preds | modelc_preds)
    none = np.sum(~modela_preds & ~modelb_preds & ~modelc_preds)
    assert anything_correct + none == len(modela)
    return anything_correct, none

if __name__ == '__main__':
    models = ['convnext_tiny', 'efficientnet_b4', 'resnet50', 'swin_tiny_patch4_window7_224', 'vit_small_patch16_224',
             'deit_small_patch16_224', 'densenet121', 'vgg11']
    models_names = ['convnext', 'effnet', 'resnet', 'swin', 'vit', 'deit', 'densenet', 'vgg']
    models_pd = []
    for model in models:
        models_pd.append(open_csv(f'topk_ids_{model}.csv'))
    
    accs = []
    for modela_id, modelb_id, modelc_id in itertools.combinations(range(len(models_pd)), 3):
        acc, none = analyze_models(models_pd[modela_id], models_pd[modelb_id], models_pd[modelc_id])
        accs.append(acc)
        print(models_names[modela_id], models_names[modelb_id], models_names[modelc_id], round(acc / 50000 * 100, 2), sep=',')
                

    print(np.sort(accs) / 50000)