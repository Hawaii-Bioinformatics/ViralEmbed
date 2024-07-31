import torch
import esm
from Bio import SeqIO
import os
from Bio.Seq import Seq
from tqdm import tqdm
from model import SparseForTokenClassification, SparseForMaskedLM
from config_model import SparseConfig
from torch import nn as nn
import urllib
import re
import http
import time
import pickle 
from Bio import Entrez

def load_fasta_as_tuples(fasta_path):
    sequences = []
    # Parcourir chaque enregistrement du fichier FASTA
    for record in SeqIO.parse(fasta_path, "fasta"):
        record_str = str(record.seq)
        record.seq = Seq(record_str)
        sequences.append((record.id, str(record.seq)))
    return(sequences)

def load_fasta_as_tuples_full(fasta_path):
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        record_str = str(record.seq)
        record.seq = Seq(record_str)
        protein = record.description.split(" ")[0]
        virus = re.search(r'\[(.*?)\]', record.description).group(1).replace(" ", ".").replace("/", ".")    # Get virus name
        if 'J' not in record_str : 
            sequences.append((virus, str(record.seq)))
    return sequences

_, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
print('alphabet loaded')

# HARDWARE
num_gpus = torch.cuda.device_count()
print(f"Nombre de GPU disponibles : {num_gpus}")

for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
#device = 'cpu'
print(device)

# DATA

#genome_fasta = '../data/results/LR760833.1.fasta'

# data = load_fasta_as_tuples(genome_fasta)


# Charger la configuration et le modèle sauvegardés
version = '3C' # or '5B'
checkpoint = torch.load(f'./models/{version}/config_and_model.pth', map_location='cpu', weights_only=False)
config = checkpoint['config']
model_state_dict = checkpoint['model_state_dict']

# Recréer le modèle avec la configuration chargée
sparse_model = SparseForTokenClassification(config=config)
sparse_model.load_state_dict(model_state_dict)
sparse_model = sparse_model.to(device)
sparse_model = sparse_model.eval()

def get_embeddings(batch_inputs,i, proteins_sizes= None, selected_proteins= None, pairwise_scores = True):
    attention_list = []
    with torch.no_grad():
        outputs = sparse_model(input_ids=batch_inputs, output_attentions = True, proteins_sizes = proteins_sizes, proteins_interactions = selected_proteins, two_step_selection=pairwise_scores)
        embeddings, attention_scores = outputs.logits, outputs.attentions
    return embeddings, attention_scores

def cumulative_sum(liste) : 
        r = 0
        res = [0]
        for l in liste : 
            r+=l
            res.append(int(r))
        return res

batch_size = 1
all_embeddings = []
all_attentions = []
all_embeddings_esm = []
all_attentions_esm = []

# saving_folder = '/home/thibaut/mahdi/saving_folder/' # For embeddings & Attentions
processed = []
unprocessed = []
names = []
all_tax = []

def extract_embeddding(embeddings, name, index):
    proteins_size = proteins_sizes_from_header(name)
    embeddings = embeddings[:,1:-1,:]
    proteins_size_cs = cumulative_sum(proteins_size)
    start, end = proteins_size_cs[index], proteins_size_cs[index+1]
    subset = embeddings[:,start:end,:]
    return subset

def extract_attention(attentions, name, index1, index2):

    proteins_size = proteins_sizes_from_header(name)
    
    proteins_size_cs = cumulative_sum(proteins_size)
    start1, end1 = proteins_size_cs[index1], proteins_size_cs[index1+1]
    start2, end2 = proteins_size_cs[index2], proteins_size_cs[index2+1]
    res = []
    for attention in attentions : 
        attention = attention[:,:,1:-1,1:-1]
        subset_r = attention[:,:,start1:end1,start2:end2]
        subset_c = attention[:,:,start2:end2,start1:end1]
        subset = subset_r + subset_c.transpose(-1,-2)
        res.append(subset)
    return res

def proteins_sizes_from_header(name) : 
    ps_list = name.split('_prots_')[1].split('_')
    ps_int = [int(k) for k in ps_list]
    return(ps_int)


# for i in tqdm(range(0, 1, batch_size)):
#     try :
#         batch_labels, batch_strs, batch_tokens = batch_converter([data[i]])
#         batch_inputs = batch_tokens
#         batch_inputs = batch_inputs.to(device)
#         sparse_model = sparse_model.to(device)
#         genome_id = data[i][0]
#         print(f'Processing genome {genome_id} ({len(data[i][0])} amino acids)')
#         ps_int = proteins_sizes_from_header(genome_id)
#         print(f'Proteins sizes = {ps_int}')
#         proteins_sizes = torch.Tensor(ps_int)
#         print('Proteins pairs ranking...')
#         embeddings, attentions_ranks = get_embeddings(batch_inputs,i, proteins_sizes= proteins_sizes, selected_proteins=None, two_step_selection=True)
#
#         print(embeddings.shape)
#         print(attentions_ranks)
#         selection = [1,6]
#         print('\n')
#         print(f'Attention pair {selection} processing...')
#         embeddings, attentions = get_embeddings(batch_inputs,i, proteins_sizes= proteins_sizes, selected_proteins=torch.Tensor(selection), two_step_selection=False)
#
#         print(embeddings.shape)
#         print(len(attentions),attentions[0].shape)
#         attentions_ag = torch.cat(attentions, dim=1)
#
#         # IF WE EXTRACT COMPLETE FRAGMENTS EMBEDDINGS & ATTENTIONS (only for 12k aa < sequences )
#
#         # extract = extract_embeddding(embeddings, genome_id, 0)
#         # print(extract.shape)
#         # extract_att = extract_attention(attentions, genome_id, 0,1)
#         # print(extract_att[0].shape)
#
#         all_embeddings.append(embeddings)
#         all_attentions.append(attentions_ag)
#         names.append(data[i][0])
#         processed.append(len(data[i][1]))
#
#     except torch.cuda.OutOfMemoryError as tc :
#         print(tc)
#
# print(f'Fragments processed = {len(all_embeddings)}')
#
# print('saving embeddings...')
# save_emb = torch.stack(all_embeddings, dim=0)
# print(save_emb.shape)
#
# torch.save(save_emb, f"{saving_folder}all_embeddings.pt")
# print(f'saved to {saving_folder}all_embeddings.pt')
#
# print('saving attentions...')
# save_att = torch.stack(all_attentions, dim=0)
# print(save_att.shape)
#
# torch.save(save_att, f"{saving_folder}all_attentions.pt")
# print(f'saved to {saving_folder}all_attentions.pt')
#
# save_names = False
# if save_names :
#     with open(f'{saving_folder}proteins_names.pkl', 'wb') as file :
#         pickle.dump(names, file)
