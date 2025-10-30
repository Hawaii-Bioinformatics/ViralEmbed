import csv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import pickle
from collections import Counter
import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_score
from tqdm import tqdm


saving_folder = ''
tax_folder = '' # Folder with taxonomic information

with open(f'{tax_folder}taxonomy_c.pkl','rb') as file : 
    taxonomy = pickle.load(file)

indexes = [1, 2, 3, 4]
results = []

for index in indexes : 
    tensor = torch.load(f'./all_embeddings.pt').squeeze(1)

    print(tensor.shape)

    taxons = []
    tax_classes = ['superkingdom', 'class', 'genus', 'species']
    print(tax_classes[index])

    for t in taxonomy : 
        try : 
            tax_dic = taxonomy[t]
            label = tax_dic[tax_classes[index]]
            if label == 'N/A' : 
                taxons.append('unclassified')
            else : 
                taxons.append(label)
        except Exception : 
            taxons.append('unclassified')
    
    top = [25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    print(top)
    sils = []
    for k in tqdm(top) : 
        raw_labels = np.array(taxons)
        tensor = tensor.cpu()
        X = tensor.numpy()

        class_counts = Counter(raw_labels)
        #valid_classes = {cls for cls, count in class_counts.items() if count >= 10 and 'unclassified' not in cls and 'unidentified' not in cls and 'non-primate' not in cls}
        top_20_elements = class_counts.most_common(k)
        top_20_set = set([element for element, frequency in top_20_elements if 'unclassified' not in element])
        valid_classes = top_20_set
        #print(valid_classes)

        filtered_data = [data for data, label in zip(X, raw_labels) if label in valid_classes]
        labels = [label for label in raw_labels if label in valid_classes]

        filtered_data = np.array(filtered_data).astype(np.float32)
        labels = np.array(labels)

        # print(filtered_data.shape)
        # print(labels.shape)

        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(filtered_data)

        silhouette_avg = silhouette_score(filtered_data, labels, metric='cosine')
        results.append(round(silhouette_avg,4))

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=labels, palette=sns.color_palette("hsv", len(set(labels))), legend='full')
    plt.text(0.95, 0.01, f'silhouette = {silhouette_avg:.3f}', horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.title('Genome embeddings - Model X - Species')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(loc='best', bbox_to_anchor=(1, 1), ncol=1)
    # plt.tight_layout()
    # plt.savefig('')
    plt.close()

print(results)
