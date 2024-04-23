import pickle
from Bio import SeqIO
from collections import defaultdict
import pandas as pd
import torch.nn  as nn
import torch
import networkx as nx
from io import StringIO
import numpy as np
from sklearn.decomposition import PCA
import sys
import torch
import numpy as np
from Bio import SeqIO, Entrez
from sklearn.cluster import KMeans
from collections import defaultdict
from functools import partial
import esm
from transformers import EsmForSequenceClassification
cos = nn.CosineSimilarity(dim=0, eps=1e-6)
top_n = 3
min_diagonal_length = 10
max_mismatches = 3

#family = str(sys.argv[1])
fasta_file = f'/Users/williamharrigan/Desktop/32_combined_sequences.fasta'
embed_dir = f'/Users/williamharrigan/Desktop/combined_will/'

family = 'pol_rnr'

#length_dict_dir = f'/home/wlh/combined_seqs_tests/fine_tune/length_dict.pkl'

def load_dict_from_pickle(filename):
    with open(filename, 'rb') as pkl_file:
        return pickle.load(pkl_file)
    
def get_seq_dict(fasta_file):
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

def get_data_matrix(seq_1, seq_2):
    x_tensor = seq_1[0]
    y_tensor = seq_2[0]

    # Normalize the vectors (this is needed for cosine similarity)
    x_norm = x_tensor / x_tensor.norm(dim=1)[:, None]
    y_norm = y_tensor / y_tensor.norm(dim=1)[:, None]

    # Compute the cosine similarity matrix
    cosine_similarity_matrix = torch.mm(x_norm, y_norm.transpose(0,1))

    # If you need the output as a DataFrame
    data = pd.DataFrame(cosine_similarity_matrix.numpy())
    return data


def find_mutual_matches_optimized(data, top_n=3):

    # Find the top_n indices for each line
    top_n_indices_rows = np.argsort(-data.values, axis=1)[:, :top_n]
    
    # Find the top_n indices for each column
    top_n_indices_cols = np.argsort(-data.values, axis=0)[:top_n, :]
    
    matches = set()
    for i in range(data.shape[0]):  # For each line
        for j in top_n_indices_rows[i]:  # For each top_n index in the line
            if i in top_n_indices_cols[:, j]:  # If row index is in column top_n
                matches.add((i, j))
                
    return matches


def add_matching_neighbors_optimized(seq_1_str, seq_2_str, matches):
    temp_set = set()

    for match in matches:
        i, j = match
        # Checking the neighbors of each match
        if i > 0 and j > 0 and seq_1_str[i - 1] == seq_2_str[j - 1]:
            temp_set.add((i - 1, j - 1))
        if i < len(seq_1_str) - 1 and j < len(seq_2_str) - 1 and seq_1_str[i + 1] == seq_2_str[j + 1]:
            temp_set.add((i + 1, j + 1))

    return matches.union(temp_set)


def find_exclusive_intervals_optimized(intervals):
    # Sort intervals by starting point, then descending end point
    intervals.sort(key=lambda x: (x[0], -x[1]))
    
    exclusive_intervals = []
    max_end_so_far = -1
    
    for interval in intervals:
        # If the end point of the current interval is greater than max_end_so_far,
        # this means that the interval is not included in any of the preceding intervals
        if interval[1] > max_end_so_far:
            exclusive_intervals.append(interval)
            max_end_so_far = interval[1]
    
    return exclusive_intervals


def find_matches_optimized(s, t, offset_val, matches, k, nb_errors=2):
    found_matches = []

    # Optimization: Run through the sequence once, keeping track of errors and matches    
    start = 0
    while start <= len(s) - k:  # Make sure there are enough characters left for a valid match
        error_count = 0
        match_length = 0
        for i in range(start, len(s)):
            # Check whether current positions match or whether a pre-existing match is recognized
            if s[i] == t[i] or (i, i + offset_val) in matches:
                match_length += 1
            else:
                error_count += 1
                if error_count > nb_errors:
                    #If the number of errors exceeds the authorized threshold, end the current check.
                    break
            
            # Check whether the current length of the valid match exceeds the threshold k
            if match_length >= k:
                found_matches.append((start, i - error_count))
                break

        start += 1  # Move to the next starting position for the next check

    # Filter the intervals found to keep only those that are exclusive
    unique_found_matches = find_exclusive_intervals_optimized(found_matches)

    return unique_found_matches


def get_matches_new(seq_1_str, seq_2_str, data, max_mismatches=3):
    matches = find_mutual_matches_optimized(data)
    matches = add_matching_neighbors_optimized(seq_1_str, seq_2_str, matches)
    valid_segments = find_all_matches_optimized(seq_1_str, seq_2_str, max_mismatches, matches)
    valid_segments = sorted(valid_segments, key=lambda x: x[0][0])
    valid_diagonals = get_valid_diagonals(valid_segments)
    matches = cleanup_matches(matches, valid_diagonals)
    
    return matches


def generate_rrotation(s, t, offset):
    """
    generate_lrotation inputs:
    s = seq_1_str
    t = seq_2_str
    offset = position in sequence where offset occurs

    generate_lrotation function rotates seq_2_str 1 position right
    along corresponding seq_1_str for each iteration and
    returns rotated string.
    """
    # If the offset is larger than the length of the
    # sequence 't', raise an exception.
    if offset >= len(s):
        raise Exception(f"offset {offset} larger than seq length {len(s)}")

    lgaps = '-' * offset

    # Extract a substring from sequence 't' starting from the offset
    # index up to the length of 's'.
    # my_str represents the part of 't' that will be kept after the rotation.
    my_str = t[0:len(s) - offset]

    # Generate a string of '-' characters of length equal to the remaining
    # length of 's' after adding 'my_str'.
    # rgaps represents the right gaps that will be added to the end of the sequence.
    rgaps = '-' * (len(s) - len(lgaps + my_str))

    return lgaps + my_str + rgaps


def generate_lrotation(s, t, offset):
    """
    generate_lrotation inputs:
    s = seq_1_str
    t = seq_2_str
    offset = position in sequence where offset occurs

    generate_lrotation function rotates seq_2_str 1 position left
    along corresponding seq_1_str for each iteration and
    returns rotated string.
    """
    # If the offset is larger than the length of the
    # sequence 't', raise an exception.
    if offset >= len(t):
        raise Exception(f"offset {offset} larger than seq length {len(s)}")

    # Extract a substring from sequence 't' starting from the offset
    # index up to the length of 's'.
    # my_str represents the part of 't' that will be kept after the rotation.
    my_str = t[offset:len(s)]

    # Generate a string of '-' characters of length equal to the remaining
    # length of 's' after adding 'my_str'.
    # rgaps represents the right gaps that will be added to the end of the sequence.
    rgaps = '-' * (len(s) - len(my_str))

    return my_str + rgaps


def find_all_matches_optimized(s, t, k, matched_pairs):
    """
    find_all_matches inputs:
    s = seq_1 sequence string denoted as 'seq_1_str'
    t = seq_2 sequence string denoted as 'seq_2_str'
    k = max_mismatches, hyperparameter defined above for amount of
    mismatches allowed.
    matched_pairs = current 'matches' list, which contains mutual matches
    and matching neighbors.
    """
    all_matches = []

    # In each iteration, generate a right rotation of 'seq_2_str' by the
    # current index and run find_match function to identify matching pairs
    # in 'seq_1_str' and 'seq_2_str' after rotation.
    # Matched pairs identified during rotation are added to all_matches
    # list.
    for i in range(0, len(s)):
        t_offset = generate_rrotation(s, t, i)

        match_in_i = find_matches_optimized(s, t_offset, -i, matched_pairs, k)

        # Adds another match along the same diagonal to match_in_i
        match_in_j = [(x - i, y - i) for x, y in match_in_i]

        # Adds both matches along same diagonal to 'all_matches' list
        all_matches.extend(list(zip(match_in_i, match_in_j)))

    # In each iteration, generate a left rotation of 'seq_2_str' by the
    # current index and run find_match function to identify matching pairs
    # in 'seq_1_str' and 'seq_2_str' after rotation.
    # Matched pairs identified during rotation are added to all_matches
    # list.
    for i in range(1, len(t)):
        t_offset = generate_lrotation(s, t, i)

        match_in_i = find_matches_optimized(s, t_offset, +i, matched_pairs, k)

        # Adds another match along the same diagonal to match_in_i
        match_in_j = [(x + i, y + i) for x, y in match_in_i]

        # Adds both matches along same diagonal to 'all_matches' list
        all_matches.extend(list(zip(match_in_i, match_in_j)))

    return all_matches


def build_paths_graph(data, matches):
    """
    build_paths_graph function identifies diagonal segments
    from sorted matches.
    """
    dag = {}

    graph = nx.DiGraph()

    max_depth = max([x[0] for x in matches])

    # Sort the matches based on the second element of the match pairs.
    sorted_matches = sorted(matches, key=lambda x: x[1])

    # Loop over the sorted matches and
    # add edges between them to build the graph.
    for i in range(len(sorted_matches) - 1):
        last_depth = max_depth
        dag[sorted_matches[i]] = []

        for j in range(i + 1, len(sorted_matches)):

            if (sorted_matches[i][0] == sorted_matches[j][0]) or (sorted_matches[i][1] == sorted_matches[j][1]):
                # Don't consider overlapping cells
                continue

            if (sorted_matches[j][0]) < last_depth and (sorted_matches[j][0] > sorted_matches[i][0]):
                dag[sorted_matches[i]].append(sorted_matches[j])
                seq_1_idx, seq_2_idx = sorted_matches[j]
                graph.add_edge(sorted_matches[i], sorted_matches[j], weigth=data.iloc[seq_1_idx, seq_2_idx])
                last_depth = sorted_matches[j][0]

    return graph


def get_valid_diagonals(valid_segments):
    """
    valid_segments = sorted(valid_segments)

    get_valid_diagonals function identifies matches that occur consecutively
    in a diagonal and stores them in a dictionary 'valid_diagonals'.
    """
    valid_diagonals = defaultdict(int)

    # Loop over the valid segments and add the length of each segment
    # to its corresponding diagonal in the dictionary.
    for x in valid_segments:
        min_val = min(x[0][0], x[1][0])
        diag = (x[0][0] - min_val, x[1][0] - min_val)
        valid_diagonals[diag] += x[0][1] - x[0][0] + 1

    return valid_diagonals


def cleanup_matches(matches, valid_diagonals):
    """
    cleanup_matches removes matches that do not occur in a valid_diagonal
    but are shorter than min_diagonal_length (hyperparameter).
    """
    remove_elems = []

    # Loop over the matches and add any invalid match to the removal list
    for x in matches:
        min_val = min(x[0], x[1])
        diag = (x[0] - min_val, x[1] - min_val)
        if valid_diagonals[diag] < min_diagonal_length:
            remove_elems.append(x)

    # Remove the invalid matches from the original list
    matches = list(set(matches).difference(remove_elems))

    return matches

def find_homologous_pos(seq_1_pos, longest_path):
    longest_path_dict = dict(longest_path)
    return longest_path_dict.get(seq_1_pos, None)

def get_longest_path(data, matches):
    longest_path = []

    # If there are any matches left, build a paths graph and find the longest path in the graph
    if len(matches) > 0:
        graph = build_paths_graph(data, matches)
        longest_path = nx.dag_longest_path(graph)

    return longest_path


def soft_align(seq_1_str, seq_2_str, seq_1_embedding, seq_2_embedding):
    data = get_data_matrix(seq_1_embedding, seq_2_embedding)
    matches = get_matches_new(seq_1_str, seq_2_str, data)
    longest_path = get_longest_path(data, matches)
    return longest_path

def get_ref_seq(seq_ids, sequence_dictionary, embed_dir):
    data = []
    ids = []
    for key, sequence in sequence_dictionary.items():
        seq_embedding = torch.load(f"{embed_dir}combined_{seq_ids.index(key)}_embeddings.pt")
        mean_embed = torch.mean(seq_embedding[0], dim=0)
        data.append(mean_embed.numpy().reshape(-1))
        ids.append(key)

    data = np.array(data)

    num_clusters = 1

    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')

    kmeans.fit(data)

    centroid = kmeans.cluster_centers_[0]

    distances = np.linalg.norm(data - centroid, axis=1)

    closest_index = np.argmin(distances)

    return ids[closest_index]


def get_matches_dict(ref_seq_id,seq_ids, sequence_dictionary, embed_dir):
    # The following code initializes a dictionary using the ref_seq
    # The dictionary consists of all of the sequence positions in the ref seq, which then serves as an index for the rest of the sequences
    # Does pairwise alingments on every pairwise combination of sequences in the sequence dictionary
    # Then all matches that occur along the longest path are stored in the dictionary under the reference seq positions
    
    ordered_prot_dict = {ref_seq_id: sequence_dictionary[ref_seq_id], **{k: v for k, v in sequence_dictionary.items() if k != ref_seq_id}}
    all_matches = defaultdict(partial(defaultdict, partial(defaultdict, partial(defaultdict, str))))
    for i in (range(len(ordered_prot_dict[ref_seq_id]))):
        all_matches[i][ref_seq_id][i]
        
    seqs = ordered_prot_dict

    for i, seq_id_1 in enumerate(list(seqs.keys())):
        seq_1_embedding = torch.load(f"{embed_dir}combined_{seq_ids.index(seq_id_1)}_embeddings.pt")
        seq_1_embedding = seq_1_embedding[:, 1:-1, :]
        seq_1_str = str(seqs[seq_id_1])
        for seq_id_2 in list(seqs.keys())[i+1:]:
            seq_2_embedding = torch.load(f"{embed_dir}combined_{seq_ids.index(seq_id_2)}_embeddings.pt")
            seq_2_embedding = seq_2_embedding[:, 1:-1, :]
            seq_2_str =  str(seqs[seq_id_2])
            longest_path = []
            longest_path = soft_align(seq_1_str, seq_2_str, seq_1_embedding, seq_2_embedding)
            for residue in all_matches.keys():
                for pos in all_matches[residue][seq_id_1].keys():
                    match_pos = find_homologous_pos(pos, longest_path)
                    if match_pos is not None:
                        all_matches[residue][seq_id_1][pos][seq_id_2] = match_pos
                        all_matches[residue][seq_id_2][match_pos][seq_id_1] = pos
                        
    # Trim the tree
    for i in all_matches.keys():
        data_wo_lowhits = all_matches[i]
        while has_multiple_sites(data_wo_lowhits):
            seq_to_rm = get_seq_w_max_diff(data_wo_lowhits)
            seq_id = list(seq_to_rm.keys())[0]
            pos_w_min = seq_to_rm[seq_id]['pos_w_min']
            if pos_w_min in data_wo_lowhits[seq_id].keys(): 
                del data_wo_lowhits[seq_id][pos_w_min]
            for k,v in data_wo_lowhits.items():
                for site,hits in v.items():
                    if seq_id in hits.keys():
                        if hits[seq_id] == pos_w_min:
                            del data_wo_lowhits[k][site][seq_id]
                            
    matches_dict = {}

    for i, data in all_matches.items():
        sorted_data = {}
        for k, v in data.items():
            sorted_items = sorted([(x, len(y)) for x, y in v.items()], key=lambda x: x[1], reverse=True)
            sorted_data[k] = sorted_items

        matches_dict[i] = sorted_data


    for key in matches_dict.keys():
        matches_dict[key] = {k: v for k, v in matches_dict[key].items() if v}
        
    return matches_dict

def has_multiple_sites(data_wo_lowhits):
    multiple_sites = False;
    for sequence in data_wo_lowhits:
        if len(data_wo_lowhits[sequence].keys()) > 1:
            multiple_sites = True
            break
    
    return multiple_sites

def get_seq_w_max_diff(data_wo_lowhits):
    all_matches_w_count = {}
    ranked = {}
    for k, v in data_wo_lowhits.items():
        all_matches_w_count[k] = {x : len(y) for x, y in v.items()}
    
    for k, v in all_matches_w_count.items():
        if len(v) > 1:
            max_value = max(v.values())
            min_value = min(v.values())
            ranked[k] = {'rank': max_value-min_value, 'pos_w_min': min(v, key=v.get)}

    ranked = {k: v for k, v in sorted(ranked.items(), key=lambda item: item[1]['rank'], reverse=True)}
    seq_to_rm = {}
    for k, v in list(ranked.items())[:1]:
        seq_to_rm = { k: v }
    
    return seq_to_rm

def contact_prediction(seq_ids, sequence_dictionary, threshold, embed_dir):
    contact_predictions = {}
    for seq_id in sequence_dictionary.keys():
        try:
            split = int(seq_id.split('_')[-1])
            seq_embedding = torch.load(f"{embed_dir}combined_{seq_ids.index(seq_id)}_full_attentions.pt")
            predictions = seq_embedding.sigmoid()
            high_values = [(i, j, value.item()) for i, row in enumerate(predictions[0]) for j, value in enumerate(row) if value.item() > threshold and i < split and j > split]
            contact_predictions[seq_id] = {(i[0], i[1]) for i in high_values}
        except:
            pass
    return contact_predictions



def predictions_in_alignments(prediction_dictionary, alignment_dictionary):
    # This code takes the contact predictions and puts them in the same index as the alignment_dictionary/matches_dict
    # Outputs dictionaries containing predictions in the matches_dict index and sequences that occur for each prediction
    
    predictions = defaultdict(list)
    sequences_in_predictions = defaultdict(list)

    for seq_id in prediction_dictionary.keys():
        for prediction in prediction_dictionary[seq_id]:
            residue_1 = None
            residue_2 = None
            for residue in alignment_dictionary.keys():
                if seq_id in alignment_dictionary[residue].keys():
                    if prediction[0] == alignment_dictionary[residue][seq_id][0][0]:
                        residue_1 = residue
                    if prediction[1] == alignment_dictionary[residue][seq_id][0][0]:
                        residue_2 = residue
            if residue_1 and residue_2:
                predictions[seq_id].append([residue_1, residue_2])
                sequences_in_predictions[residue_1, residue_2].append(seq_id)
                
    return predictions, sequences_in_predictions


def ft_pipeline(family, fasta_file, embed_dir):
 #   length_dict =  load_dict_from_pickle(length_dict_dir)
 #   length_dict = length_dict[family]
    sequence_dict = dict(list(get_seq_dict(fasta_file).items()))
    seq_ids = list(sequence_dict.keys())
    ref_seq_id = get_ref_seq(seq_ids, sequence_dict, embed_dir)
    matches_dict = get_matches_dict(ref_seq_id, seq_ids, sequence_dict, embed_dir)
    contacts = contact_prediction(seq_ids, sequence_dict, 0.95, embed_dir)
    predictions, sequences_in_predictions = predictions_in_alignments(contacts, matches_dict)

    with open(f'{family}_matches_dict.pkl', 'wb') as f:
        pickle.dump(matches_dict, f)
    with open(f'{family}_contacts.pkl', 'wb') as f:
        pickle.dump(contacts, f)
    with open(f'{family}_predictions.pkl', 'wb') as f:
        pickle.dump(predictions, f)
    with open(f'{family}_sequences_in_predictions.pkl', 'wb') as f:
        pickle.dump(sequences_in_predictions, f)

    return 'done'

#ft_pipeline(family, fasta_file, embed_dir)
