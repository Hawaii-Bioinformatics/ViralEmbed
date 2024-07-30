# Full script for pair ranking and visualisation - input = attentions + annotated fasta file + folder annotated fragments (get_annotations.py)

import torch
from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np
import os
from Bio.Seq import Seq
from tqdm import tqdm
import torch
from collections import defaultdict
import seaborn as sns
import pickle
import networkx as nx
import time as t 
from multiprocessing import Pool, Manager, cpu_count, set_start_method

# FASTA FILE PATH

# Fasta of genomes -> labels should be genome_id_prots_n1_n2_...
hel_pol_rnr_assembled_fasta = 'data/sample_genome.fasta'
# Attentions scores path : should be "pol_hel_rnr_{i}_full_attention_weighted.pt" with i index in fasta file. 
attention_file = 'data/LR699048.pt'
# Proteins labels path (used for plot, to fix).
labels_file = 'data/LR699048.txt'
# For versions
saving_id = '3'
# Saving folder single fragment heatmap 
single_heatmap_path = 'data/results'
# Path to save as pkl interaction scores
pkl_save_path = 'data/results'


run = True
plot_figures = False
multiprocess = False

import torch


def apc(x):
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)
    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    x.sub_(avg) # in-place to reduce memory
    del avg
    return x

def get_attention_block(self_attention_t_weighted, p1_boundaries, p2_boundaries, apc_norm=False):
    """
        self_attention: square matrix representing the self attentions
        returns: mutal attention between p1 and p2        
    """
    block1 = self_attention_t_weighted[0][p1_boundaries[0]: p1_boundaries[1], p2_boundaries[0]: p2_boundaries[1]]
    block2 = self_attention_t_weighted[0][p2_boundaries[0]: p2_boundaries[1], p1_boundaries[0]: p1_boundaries[1]]

    block2_transposed = block2.t()
    mutual_information_tensor = block1 + block2_transposed

    if apc_norm : 
        mutual_apc = apc((mutual_information_tensor))  
        return mutual_apc  
    else :     
        return mutual_information_tensor

def get_contacts_numbers(prot1, prot2, attention_block, plot = False):
    attentions = np.array(attention_block.reshape(1, -1).tolist()[0])
    list_attn = [((prot1, prot2), len(attentions), att_item) for i, att_item in enumerate(attentions)]
    return list_attn

def get_contacts_numbers_torch(prot1, prot2, attention_block):
    attentions = attention_block.view(-1)
    size = attentions.size(0)
    contacts = [(prot1, prot2)] * size
    list_attn = list(zip(contacts, [size] * size, attentions.tolist()))
    return list_attn

def get_pairs(length, segment_attentions) : 
    # Get all the amino acid pairs and attention score of a fragment
    # Length is the list of 
    all_pairs = []
    for i in range(len(length)-1) : # Over proteins
        for j in range(i+1, len(length)) : # Over proteins
            p1_start, p1_end, p2_start, p2_end = sum(length[:i]), sum(length[:i+1]),  sum(length[:j]), sum(length[:j+1])    # Get starting and ending amino acids index for protein pair
            attention_block = get_attention_block(segment_attentions, (p1_start, p1_end), (p2_start, p2_end), apc_norm=False)    # Extract attention block
            results = get_contacts_numbers_torch(i, j, attention_block) # Get the list [[(p1, p2), n1*n2, att] for each aa pair]
            all_pairs.extend(results)
    return(all_pairs)

def load_fasta_as_tuples(fasta_path):
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        record_str = str(record.seq)
        record.seq = Seq(record_str)
        sequences.append((record.id,str(record.seq)))
    return sequences

def load_fasta_as_dic(fasta_path):
    sequences = {}
    # Parcourir chaque enregistrement du fichier FASTA
    for record in SeqIO.parse(fasta_path, "fasta"):
        record_str = str(record.seq)
        record.seq = Seq(record_str)
        sequences[record.id] = str(record.seq)
    return sequences

def rank_assembled_pairs(id, attention_matrix, annotation_dic) :
    t0t = t.time()
    if len(id.split('_prots_')[1]) == 0 : 
        return None, None
    
    length_frag = list(map(int, id.split('_prots_')[1].split('_')))

    if len(length_frag) == 1 :
        return None, None
    
    first_prot_frag = 0
    pairs_frag = get_pairs(length_frag, attention_matrix)   # Every amino acid pair, attention value : ((p1, p2), n1*n2, att)
   
    #sorted_data_frag = sorted(pairs_frag, key=lambda x: x[2], reverse=True) # Sort by attention value
    #top_data_frag = sorted_data_frag[:int(0.02*len(sorted_data_frag))]  # Keep only 2% with highest score
    
    atts = [item[2] for item in pairs_frag]
    p2 = np.percentile(atts, 2)
    top_data_frag = [item for item in pairs_frag if item[2] < p2 ]
    

    pair_counts_frag = defaultdict(int) # New dic {(p1,p2):score}
    pair_index_sums_frag = defaultdict(int) # New dic {(p1,p2):n1*n2}
    attentions_scores_frag = [x[2] for x in top_data_frag]  # All attentions scores

    a, b = min(attentions_scores_frag), max(attentions_scores_frag)  

    for pair, index_value, score in top_data_frag :
        pair_counts_frag[pair] += (score-a)/(b-a)   # Fill dic (score to reduce threshold impact)
        pair_index_sums_frag[pair] = index_value    # Fill dic with block size
    

    pair_normalizations_frag = {pair: count / pair_index_sums_frag[pair] for pair, count in pair_counts_frag.items()}   # Final dictionnary {(p1,p2) : score normalized by size}
    top_pairs_frag = sorted(pair_normalizations_frag.items(), key=lambda item: item[1], reverse=True)   # Sort by score
    all_pairs_ranked = []
    non_consecutive_pairs_ranked = []
    i,j = 0,0

    # Filter consecutive pairs
    for pair, normalization in top_pairs_frag:
        all_pairs_ranked.append(pair)

        if abs(pair[0]-pair[1]) != 1 : 
            non_consecutive_pairs_ranked.append(pair)

    # For annotations
    top_pairs_frag_nc = []

    for pair, normalization in top_pairs_frag:
        if abs(pair[0]-pair[1]) != 1 : 
            top_pairs_frag_nc.append((pair, normalization))

    return all_pairs_ranked , non_consecutive_pairs_ranked, top_pairs_frag, top_pairs_frag_nc

def list_to_dict(pairs_list):
    it = iter(pairs_list)
    names_dict = dict(zip(it, it))

    result_dict = {}
    keys = list(names_dict.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key_pair = (int(names_dict[keys[i]]), int(names_dict[keys[j]]))
            value_pair = (keys[i], keys[j])
            result_dict[key_pair] = value_pair

    return result_dict

def merge_dict_scores(input_dict):
    merged_dict = dict(input_dict)

    for key in list(merged_dict.keys()):
        if not merged_dict[key]:  
            inverse_key = tuple(reversed(key))
            if merged_dict.get(inverse_key):  
                merged_dict[key] = merged_dict[inverse_key]

    final_dict = {}
    for key, value in merged_dict.items():
        sorted_key = tuple(sorted(key))
        if sorted_key not in final_dict:  
            final_dict[sorted_key] = value

    return final_dict

def sum_dict(dic1, dic2) : 
    result_dic = dic1
    for k, v in dic1.items() : 
        summed_tuple = tuple(a + b for a, b in zip(v[0], dic2[k][0]))
        result_dic[k] = [summed_tuple]
    return result_dic

def insert_newlines(label, words_per_line=20):
    words = label.split()
    lines = [' '.join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    return '\n'.join(lines)

def single_fragment_heatmap(labels_file, top_pairs_frag, genome_id) : 
    with open(labels_file, 'r') as file:
        labels = [line.strip() for line in file.readlines()]

    n = len(labels)
    matrix = np.zeros((n, n))

    if n < 15 : 
        adjusted_labels = [insert_newlines(label) for label in labels]
    else : 
        adjusted_labels = labels

    for (i, j), value in top_pairs_frag:
        matrix[i, j] = value
        matrix[j, i] = value*100000  
        #annotated_all = ((i,j), value, (labels[i], labels[j]))
        # all_top_pairs.append(annotated_all)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=adjusted_labels, yticklabels=adjusted_labels, annot_kws={"size": 8})
    plt.title("Attention heatmap")
    plt.tight_layout()
    plt.savefig(f'{single_heatmap_path}{genome_id}_heatmap.png')
    
def add_label(labels_file, top_pairs_frag, genome_id) : 

    with open(labels_file, 'r') as file:
        labels = [line.strip() for line in file.readlines()]
    ans = []
    for (i, j), value in top_pairs_frag:
        ans.append(((labels[i], labels[j]),value))

    return(ans)

def get_pairs_dataset(fasta, attention_file, labels_file, plot_fragment_heatmap = False) :
    """
    returns attention scores for all pairs in a dataset
    """
    
    data_assembled = load_fasta_as_tuples(fasta)
    print(f"{len(data_assembled)} assembled fragments" )

    # Global lists for saving data
    global_length = []
    non_consecutive_length = []
    all_length_dict = []
    j = 0
    all_top_pairs = []
    all_top_pairs_nc = []
    all_top_pairs_labeled = []
    all_top_pairs_labeled_nc = []
    database_dic = {}
    mlp = []
    genomes = []

    plot_fragment_heatmap = False
    for i, tuples in enumerate(data_assembled) : 
        if os.path.exists(attention_file) :  # Check for attention file
            try : 
                id = tuples[0]
                genome_id = id.split('_')[0]
                genomes.append(genome_id)
                id_list = id.split('_prots_')[0].split('_')[1:]
                annotation_dic = list_to_dict(id_list)  # From list of annotated proteins, return list of pairs with their annotation : ['HEL', '0', 'RNR', '4', 'POLA', '6'] -> {(0, 4): ('HEL', 'RNR'), (0, 6): ('HEL', 'POLA'), (4, 6): ('RNR', 'POLA')}

                attention_block = torch.load(attention_file, weights_only=True) # Load attention file

                if multiprocess : 
                    mlp.append((id, attention_block, annotation_dic))

                else : 
                    all_pairs_ranked, non_consecutive_pairs_ranked, top_pairs_frag, top_pairs_frag_nc = rank_assembled_pairs(id, attention_block, annotation_dic)    # Get ranked protein pairs and scores
                    all_top_pairs.append(top_pairs_frag)    # Store all pairs
                    all_top_pairs_nc.append(top_pairs_frag_nc)  # Store all non consecutive pairs

                    j+=1
                    global_length.append(len(all_pairs_ranked))     # Store number of pairs
                    non_consecutive_length.append(len(non_consecutive_pairs_ranked))    # Store number of non consecutive pairs

                    # labels_file = labels_file  # Get labels for heatmap & labeled lists
                    database_dic[genome_id] = top_pairs_frag_nc

                    if plot_fragment_heatmap : 
                        single_fragment_heatmap(labels_file, top_pairs_frag, genome_id)

                    top_pairs_frag_labeled = add_label(labels_file, top_pairs_frag, genome_id)
                    top_pairs_frag_labeled_nc = add_label(labels_file, top_pairs_frag_nc, genome_id)

                    all_top_pairs_labeled.append(top_pairs_frag_labeled)    
                    all_top_pairs_labeled_nc.append(top_pairs_frag_labeled_nc) 
                    
            except KeyError as k : 
                print(f"KeyError for {id}")

    if not multiprocess : 
        print(f"{j} fragments processed")
        return all_top_pairs, all_top_pairs_nc, global_length, non_consecutive_length, database_dic, all_top_pairs_labeled , all_top_pairs_labeled_nc

    if multiprocess : 
        print(f"Number of sequences to process : {len(mlp)}")

        ### Multiprocessing
        nombre_cpus = cpu_count()
        print(f'{nombre_cpus} cpus availables')

        with Pool(processes=nombre_cpus//5) as pool:
            results = pool.starmap(rank_assembled_pairs, mlp)

        print(f"{len(mlp)} fragments processed")

        all_top_pairs = []
        all_top_pairs_nc = []
        all_top_pairs_labeled = []
        all_top_pairs_labeled_nc = []
        global_length = []
        non_consecutive_length = []
        database_dic_mlp = {}
        
        for k, (all_pairs_ranked, non_consecutive_pairs_ranked, top_pairs_frag, top_pairs_frag_nc) in enumerate(results) :
            genome_id = genomes[k] 
            all_top_pairs.append(top_pairs_frag)    # Store all pairs
            all_top_pairs_nc.append(top_pairs_frag_nc)  # Store all non consecutive pairs
            global_length.append(len(all_pairs_ranked))
            non_consecutive_length.append(len(non_consecutive_pairs_ranked))
            database_dic_mlp[genome_id] = top_pairs_frag_nc

            # labels_file = f'{labels_folder}{genome_id}.txt'
            top_pairs_frag_labeled = add_label(labels_file, top_pairs_frag, genome_id)
            top_pairs_frag_labeled_nc = add_label(labels_file, top_pairs_frag_nc, genome_id)

            all_top_pairs_labeled.append(top_pairs_frag_labeled)    
            all_top_pairs_labeled_nc.append(top_pairs_frag_labeled_nc) 

        return all_top_pairs, all_top_pairs_nc, global_length, non_consecutive_length, database_dic,all_top_pairs_labeled , all_top_pairs_labeled_nc

def get_results_dic(all_top_pairs, common_list) :
    """
    converts retults to dict
    """
    results = {}
    for genome in all_top_pairs :
        for item in genome : 
            key = item[0]  
            score = item[1]  
            if key in results and (key[0] not in common_list) and (key[1] not in common_list):
                results[key] = (results[key][0] + 1, results[key][1] + score)
            else:
                results[key] = (1, score)
    return results 

def norm_results(results_nc, threshold = 1) : 
    results_norm_nc = {}
    for key, value in results_nc.items():
        if value[0] >= threshold : 
            results_norm_nc[key] = value[1]/value[0]
    return results_norm_nc

def get_top_pairs(results, i = 20) : 
    """ Get the i pairs with the highest scores - before making a network / heatmap """ 
    top_interactions = sorted(results.items(), key=lambda x: x[1], reverse=True)[:i]
    return top_interactions

def plot_interaction_graph(top_interactions, save_path = None, consecutive = False) : 

    # Data
    G = nx.Graph()
    for (node1, node2), value in top_interactions:
        weight = value
        G.add_edge(node1, node2, weight=weight)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    edge_widths = [G[u][v]['weight']*20 for u, v in G.edges()]

    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, edge_color='lightgray', width=edge_widths)
    if consecutive : 
        plt.title("Protein cluster interactions - consecutive included - 50 top interactions")
    else : 
        plt.title("Protein cluster interactions - non-consecutive - 20 top interactions")

    if save_path == None :      
        plt.show()
    else : 
        plt.savefig(save_path)

def combined_heatmap(top_interactions, save_path = None, consecutive = False): 

    clusters = []
    matrix_idx = {}

    for pair, _ in top_interactions:
        if pair[0] not in clusters:
            clusters.append(pair[0])
            matrix_idx[pair[0]] = len(clusters) - 1
        if pair[1] not in clusters:
            clusters.append(pair[1])
            matrix_idx[pair[1]] = len(clusters) - 1
        
    n = len(clusters)
    matrix_nc = np.zeros((n, n))

    # Remplir la matrice avec vos données
    for (i, j), value in [((matrix_idx[pair[0]], matrix_idx[pair[1]]), score) for pair, score in top_interactions]:
        matrix_nc[i, j] = value
        matrix_nc[j, i] = value  

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_nc, annot=True, fmt=".1f", cmap="coolwarm", xticklabels=clusters, yticklabels=clusters, annot_kws={"size": 6})
    if consecutive : 
        plt.title("Attention heatmap - consecutive pairs included")
    else : 
        plt.title("Attention heatmap - only non-consecutive pairs")

    if save_path == None :      
        plt.show()
    else : 
        plt.savefig(save_path)


# if run :
#     all_top_pairs, all_top_pairs_nc, global_length, non_consecutive_length, database_dic, all_top_pairs_labeled , all_top_pairs_labeled_nc = get_pairs_dataset(hel_pol_rnr_assembled_fasta, attention_file, labels_file)
#
#     # Save interactions in a pickle file to avoid all the code above
#     with open(pkl_save_path+f'all_top_pairs_nc_{saving_id}.pkl', 'wb') as f:
#         pickle.dump(all_top_pairs_nc, f)
#
#     with open(pkl_save_path+f'all_top_pairs_{saving_id}.pkl', 'wb') as f:
#         pickle.dump(all_top_pairs, f)
#
#     with open(pkl_save_path+f'database_dic_{saving_id}.pkl', 'wb') as f:
#         pickle.dump(database_dic, f)
#
#     print(f"Pairs saved in {pkl_save_path}")
#
# else :
#     # Load existing interaction pickle files
#     with open(pkl_save_path+'all_top_pairs_nc.pkl', 'rb') as f:
#         all_top_pairs_nc = pickle.load(f)
#
#     with open(pkl_save_path+'all_top_pairs.pkl', 'rb') as f:
#         all_top_pairs = pickle.load(f)


# # IF Labels are used
# labels = True     # TOFIX
#
# if labels :
#     common_list = 'Phage protein'
#     results = get_results_dic(all_top_pairs_labeled, common_list)
#     results_norm = norm_results(results, threshold = 1)             # threshold = minimum number of apparitions across genomes
#     top_interactions = get_top_pairs(results_norm, i = 20)          # i = Number of pairs
#
#     results_nc = get_results_dic(all_top_pairs_labeled_nc, common_list)
#     results_norm_nc = norm_results(results_nc, threshold = 1)       # threshold = minimum number of apparitions across genomes
#     top_interactions_nc = get_top_pairs(results_norm_nc, i = 20)    # i = Number of pairs
#
#
# if plot_figures :
#     # NX chart for interactions
#     plot_interaction_graph(top_interactions, pkl_save_path+f'20_interactions_nx_{saving_id}.png', consecutive = True)
#     plot_interaction_graph(top_interactions_nc,  pkl_save_path+f'20_nc_interactions_nx_{saving_id}.png')
#
#     # Heatmap (1 block = 1 cluster)
#     combined_heatmap(top_interactions, pkl_save_path+f'20_interactions_sns_{saving_id}.png', consecutive = True)
#     combined_heatmap(top_interactions_nc,  pkl_save_path+f'20_nc_interactions_sns_{saving_id}.png')
#
# else :
#     print(top_interactions_nc)
#     print(top_interactions)