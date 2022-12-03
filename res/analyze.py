import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


print(os.listdir())

def open_csv(path):
    df = pd.read_csv(path, names=['file', 'predicted', 'truth'])
    return df

def analyze_models(modela, modelb):
    both_correct = 0
    a_correct = 0
    b_correct = 0
    both_incorrect = 0

    assert (modela['file'] == modelb['file']).to_numpy().all()
    modela_preds = (modela['predicted'] == modela['truth']).to_numpy()
    modelb_preds = (modelb['predicted'] == modelb['truth']).to_numpy()

    both_correct = np.sum(modela_preds & modelb_preds)
    a_correct = np.sum(modela_preds & ~modelb_preds)
    b_correct = np.sum(~modela_preds & modelb_preds)
    both_incorrect = np.sum(~modela_preds & ~modelb_preds)
    return both_correct, a_correct, b_correct, both_incorrect

if __name__ == '__main__':
    models = ['convnext_tiny', 'efficientnet_b4', 'resnet50', 'swin_tiny_patch4_window7_224', 'vit_small_patch16_224',
             'deit_small_patch16_224', 'densenet121', 'vgg11'] # 'convit_small', 'convmixer_768_32'
    models_names = ['ConvNeXt', 'EffNet', 'ResNet', 'Swin', 'ViT', 'DeiT', 'DenseNet', 'VGG'] # 'ConVit', 'ConvMixer'
    models_pd = []
    for model in models:
        models_pd.append(open_csv(f'topk_ids_{model}.csv'))
    sns.set(font_scale=10)
    plt.rcParams.update({'font.size': 50})

    total_imgs = 50000
    accs = []
    fixes = []
    for modela in models_pd:
        accs.append([])
        fixes.append([])
        for modelb in models_pd:
            both_correct, a_correct, b_correct, both_incorrect = analyze_models(modela, modelb)
            accs[-1].append(100 * (both_correct + a_correct + b_correct) / total_imgs)
            a_incorrect = total_imgs - both_correct - a_correct
            fixes[-1].append(100 * (b_correct / a_incorrect))
    mask = np.zeros_like(accs)
    # mask[np.triu_indices_from(mask)] = True
    # mask[np.diag_indices_from(mask)] = False
    plt.rcParams.update({'font.size': 22})
    sns.set(rc = {'figure.figsize':(15,8)})

    fig = sns.heatmap(accs, annot=True, fmt='.2f', mask=mask, cmap='flare',
                    xticklabels=models_names, 
                    yticklabels=models_names)
    fig.set_xlabel(xlabel='Second Model (Model that corrects the first model)', ylabel='First Model (Base Model)')
    plt.tight_layout()
    fig.get_figure().savefig('accs2.png')
    fig.get_figure().savefig('accs2.pdf')
    plt.clf()
    print(fixes)
    fig = sns.heatmap(fixes, annot=True, fmt='.2f', mask=mask, cmap='flare',
                    xticklabels=models_names, 
                    yticklabels=models_names)
    fig.set(xlabel='Second Model (Model that corrects the first model)', ylabel='First Model (Base Model)')
    plt.tight_layout()
    fig.get_figure().savefig('fixes.png')
    fig.get_figure().savefig('fixes.pdf')