# Script for complete pair ranking, and saving in pkl file


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
from multiprocessing import Pool, Manager, cpu_count, set_start_method
import pickle 

hel_pol_rnr_assembled_fasta = '/home/thibaut/Keep_assembled_annotated_sequences_corr_annot_11k.fasta'
attention_path = '/home/thibaut/KEEP_80_v3_results/attentions/'
pkl_save_path = '/home/thibaut/KEEP_80_v3/pkl/'
labels_folder = '/home/thibaut/KEEP_80_v3/annots/'
saving_id = '3'
    

run = True
plot_figures = False


def apc(x):
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)
    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    x.sub_(avg) # in-place to reduce memory
    del avg
    return x

def get_attention_block(self_attention_t_weighted, p1_boundaries, p2_boundaries, apc_norm=True):
    """
        self_attention: square matrix representing the self attentions
        returns: mutal attention between p1 and p2        
    """
    block1 = self_attention_t_weighted[0][0][p1_boundaries[0]: p1_boundaries[1], p2_boundaries[0]: p2_boundaries[1]]
    block2 = self_attention_t_weighted[0][0][p2_boundaries[0]: p2_boundaries[1], p1_boundaries[0]: p1_boundaries[1]]
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

def get_pairs(length, segment_attentions) : 
    all_pairs = []
    for i in range(len(length)-1) : 
        for j in range(i+1, len(length)) : 
            p1_start, p1_end, p2_start, p2_end = sum(length[:i]), sum(length[:i+1]),  sum(length[:j]), sum(length[:j+1])
            attention_block = get_attention_block(segment_attentions, (p1_start, p1_end), (p2_start, p2_end), apc_norm=True)
            results = get_contacts_numbers(i, j, attention_block, plot = False)
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

def create_genom_dic(fasta) : 
    genome_dict = {}

    for record in SeqIO.parse(fasta, "fasta"):
        protein_id = record.id
        genome_id = protein_id.split('_')[1]  

        if genome_id not in genome_dict:

            genome_dict[genome_id] = []

        genome_dict[genome_id].append(protein_id)

    return genome_dict

def rank_assembled_pairs(id, attention_matrix, annotation_dic, process_ranks = False, print_pairs = False) :
    print(id.split('_prots_')[0])
    if len(id.split('_prots_')[1]) == 0 : 
        return None, None
    
    genom_dic = create_genom_dic('/home/thibaut/Keep_11k_proteins.fasta')
    x = id.split('_')[0]
    
    length_frag = list(map(int, id.split('_prots_')[1].split('_')))

    if len(length_frag) == 1 :
        return None, None
    
    first_prot_frag = 0
    pairs_frag = get_pairs(length_frag, attention_matrix)
    sorted_data_frag = sorted(pairs_frag, key=lambda x: x[2], reverse=True)
    
    top_data_frag = sorted_data_frag[:int(0.02*len(sorted_data_frag))]
    uniq_top_pairs = list(set([pair[0] for pair in top_data_frag]))
    # print(uniq_top_pairs)

    dictionnaire = {cle: 0 for cle in uniq_top_pairs}
    for pair in top_data_frag : 
        dictionnaire[pair[0]] += pair[2]/pair[1]

    dictionnaire2 = {}
    for k, v in dictionnaire.items() : 
        dictionnaire2[(genom_dic[x][k[0]], genom_dic[x][k[1]])] = v
    
    with open(f'{pkl_save_path}{x}.pkl', 'wb') as f:
        pickle.dump(dictionnaire2, f)
    
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
    
def get_pairs_dataset(fasta, attention_path, labels_folder, process_ranks = False, print_pairs = False, plot_fragment_heatmap = False) : 
    
    data_assembled = load_fasta_as_tuples(fasta)
    print(f"{len(data_assembled)} assembled fragments" )

    # rank_assembled_pairs parameters
    process_ranks = False
    print_pairs = False

    # Global list for saving data
    mlp = []

    for i, tuples in tqdm(enumerate(data_assembled)) : 
        if os.path.exists(attention_path+f'full_{i}.pt') : 
            # try : 
                id = tuples[0]
                id_list = id.split('_prots_')[0].split('_')[1:]
                annotation_dic = list_to_dict(id_list)

                attention_block = torch.load(attention_path+f'full_{i}.pt', map_location=torch.device('cpu'))
                mlp.append((id, attention_block, annotation_dic, process_ranks, print_pairs))
                
    print(f"Number of sequences to process : {len(mlp)}")

    ### Multiprocessing
    nombre_cpus = cpu_count()
    print(f'{nombre_cpus} cpus availables')

    with Pool(processes=nombre_cpus//5) as pool:
        results = pool.starmap(rank_assembled_pairs, mlp)
    

if run : 
    
    all_top_pairs, all_top_pairs_nc, global_length, non_consecutive_length, database_dic = get_pairs_dataset(hel_pol_rnr_assembled_fasta, attention_path, labels_folder)
    
