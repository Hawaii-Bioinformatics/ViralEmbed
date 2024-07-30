import torch
import numpy as np
from collections import defaultdict


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
    block1 = self_attention_t_weighted[0][int(p1_boundaries[0]): int(p1_boundaries[1]), int(p2_boundaries[0]): int(p2_boundaries[1])]
    block2 = self_attention_t_weighted[0][int(p2_boundaries[0]): int(p2_boundaries[1]), int(p1_boundaries[0]): int(p1_boundaries[1])]

    block2_transposed = block2.t()
    mutual_information_tensor = block1 + block2_transposed

    if apc_norm : 
        mutual_apc = apc((mutual_information_tensor))  
        return mutual_apc  
    else :     
        return mutual_information_tensor

def get_contacts_numbers_torch(prot1, prot2, attention_block):
    attentions = attention_block.view(-1)
    size = attentions.size(0)
    contacts = [(prot1, prot2)] * size
    list_attn = list(zip(contacts, [size] * size, attentions.tolist()))
    return list_attn

def get_pairs(length, segment_attentions) : 
    # Get all the amino acid pairs and attention score of a fragment
    # Length is the list of proteins sizes
    all_pairs = []
    for i in range(len(length)-1) : # Over proteins
        for j in range(i+1, len(length)) : # Over proteins
            p1_start, p1_end, p2_start, p2_end = sum(length[:i]), sum(length[:i+1]),  sum(length[:j]), sum(length[:j+1])    # Get starting and ending amino acids index for protein pair
            attention_block = get_attention_block(segment_attentions, (p1_start, p1_end), (p2_start, p2_end), apc_norm=True)    # Extract attention block
            results = get_contacts_numbers_torch(i, j, attention_block) # Get the list [[(p1, p2), n1*n2, att] for each aa pair]
            all_pairs.extend(results)
    return(all_pairs)

def rank_assembled_pairs(proteins_sizes, attention_matrix) :
    
    pairs_frag = get_pairs(proteins_sizes, attention_matrix)   # Every amino acid pair, attention value : ((p1, p2), n1*n2, att)
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
    all_pairs_ranked = sorted(pair_normalizations_frag.items(), key=lambda item: item[1], reverse=True)   # Sort by score
    all_pairs_ranked_dic = {}

    for pair, normalization in all_pairs_ranked:
        all_pairs_ranked_dic[pair] = normalization

    return all_pairs_ranked, all_pairs_ranked_dic
