import torch
import os 
import pickle
from tqdm import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

base_folder = './embeds_3C'
folder = './embeds_3C_V3/attentions'
print(base_folder)

def mean(liste) :
    return(sum(liste)/len(liste))

def best_head(stat_list, index) : 
    ratio_list = []
    for h in stat_list :
        mean = sum(h)/len(h)
        ratio = h[index]/mean
        ratio_list.append(ratio)
    
    max = (0, ratio_list[0])
    for i, r in enumerate(ratio_list) : 
        if r > max[1] : 
            max = (i, r)

    return max, ratio_list

def sort_dic_values(dic, reversed = True) : 
    dictionnaire_trie = dict(sorted(dic.items(), key=lambda item: item[1], reverse=reversed))
    return dictionnaire_trie

ct0, ct1, ct2 = 0, 0, 0
st0, st1, st2 = [], [], []
best_dic = {}
all_dic = {}

for k, file in tqdm(enumerate(os.listdir(folder))) : 
    file_path = os.path.join(folder, file)
    genome_name = file_path[6:-4]
    with open(file_path, 'rb') as f : 
        stats = pickle.load(f)
    print(len(stats), len(stats[0]))
    if math.isnan(stats[0][2]): 
        continue

    t0 = [head[0] for head in stats]
    t1 = [head[1] for head in stats]
    t2 = [head[2] for head in stats]

    mt0 = sum(t0)/len(t0)
    st0.append(mt0)
    mt1 = sum(t1)/len(t1)
    st1.append(mt1)
    mt2 = sum(t2)/len(t2)
    st2.append(mt2)

    if max((mt0, mt1, mt2)) == mt0 : 
        #print(f'Max = mt0 = {mt0}')
        ct0 +=1
    elif max((mt0, mt1, mt2)) == mt1 : 
        #print(f'Max = mt1 = {mt1}')
        ct1 +=1
    elif max((mt0, mt1, mt2)) == mt2 : 
        #print(f'Max = mt2 = {mt2}')
        ct2 +=1

    max_head, ratio_list = best_head(stats, 2) 

    if max_head[0] in best_dic : 
        best_dic[max_head[0]]+=1
    else :
        best_dic[max_head[0]]=1
    
    for i, ratio in enumerate(ratio_list) : 
        if k == 0 :
            all_dic[i] = ratio
        else : 
            all_dic[i] += ratio

#print(sort_dic_values(all_dic))
print(sort_dic_values(best_dic))

print(f'ct0 = {ct0}')
print(f'ct1 = {ct1}')
print(f'ct2 = {ct2}')

ti_list = [mean(st0), mean(st1), mean(st2)]

for i, mt in enumerate(ti_list) : 
    print(f'mt{i} = {mt/sum(ti_list)*100:.4f}%')


figure_heads = True
if figure_heads : 

    num_heads = 20
    num_layers = len(all_dic)//num_heads                         # TO CHANGE FOR 33 LAYERS
    data_matrix = np.zeros((num_layers, num_heads))

    for key, value in all_dic.items():
        layer = key // num_heads
        head = key % num_heads
        if layer < num_layers:
            data_matrix[layer, head] = value


    colors = [(0.7, 0.9, 1), (0, 0, 1)] 
    n_bins = 100  
    cmap_name = 'custom_blue'
    cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # MAIN FIGURE
    fig, ax = plt.subplots(figsize=(10,10))

    cax = ax.imshow(data_matrix, cmap='BuGn', aspect='auto', interpolation='none')

    ax.set_xticks(np.arange(data_matrix.shape[1]))
    ax.set_yticks(np.arange(data_matrix.shape[0]))
    ax.set_xticklabels(np.arange(data_matrix.shape[1]), fontsize = 12)
    ax.set_yticklabels(np.arange(data_matrix.shape[0]), fontsize = 12)

    ax.set_xticks(np.arange(-0.5, data_matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data_matrix.shape[0], 1) - 0.02, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Heads', fontsize = 14)
    ax.set_ylabel('Layers', fontsize = 14)
    #ax.set_title(False)

    cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label('Long range importance value (over 500 amino acid)', fontsize = 14)

    plt.savefig('head.png', dpi=1000)
    plt.close()

