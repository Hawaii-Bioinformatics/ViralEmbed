import torch
import esm
from Bio import SeqIO
# import os
from Bio.Seq import Seq
from tqdm import tqdm
from model import SparseForTokenClassification, SparseForMaskedLM
# from config_model import SparseConfig
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

# #fasta_file = '/home/thibaut/LongVirus/data/viral.1.protein.faa'
# genom_fasta = 'data/LR699048.fa'
# data = load_fasta_as_tuples(genom_fasta)
# num_genoms = len(data)
# data = [genome for genome in genoms if len(genome[1])<14000 and len(genome[1])>13000]
# genomes_name = [genome[0] for genome in genoms if len(genome[1])>61000]


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


def stat_attention(attentions, t1, t2):
    ans = []
    head = attentions[0]
    n = attentions.shape[2]

    idx = torch.arange(n)
    idx_i, idx_j = torch.meshgrid(idx, idx)
    
    abs_diff = (idx_i - idx_j).abs()
    
    mask_t1 = abs_diff < t1
    mask_t2 = (abs_diff >= t1) & (abs_diff < t2)
    mask_t3 = abs_diff >= t2

    for head in attentions[0]:
        t1_values = head[mask_t1]
        t2_values = head[mask_t2]
        t3_values = head[mask_t3]

        run = (
            t1_values.mean().item(),
            t2_values.mean().item(),
            t3_values.mean().item()
        )
        ans.append(run)
    return ans

def get_embeddings(batch_inputs,i, proteins_sizes= None, selected_protein= None):
    attention_list = []
    with torch.no_grad():
        outputs = sparse_model(input_ids=batch_inputs, output_attentions = True, proteins_sizes = proteins_sizes, proteins_interactions = selected_protein)
        embeddings, attention_scores = outputs.logits, outputs.attentions
    return embeddings, attention_scores


# batch_size = 1
# all_embeddings = []
# all_attentions = []
# all_embeddings_esm = []
# all_attentions_esm = []
#
# saving_folder = 'data/results/inference'
# processed = []
# unprocessed = []
# names = []
# all_tax = []


# def get_taxonomy_from_genome_name(genome_name, email):
#     """
#     Récupère la taxonomie pour un nom de génome donné.
#
#     :param genome_name: str, le nom du génome (par exemple, "Human papillomavirus 135")
#     :param email: str, votre adresse email (nécessaire pour accéder à l'API NCBI)
#     :return: list, la taxonomie sous forme de liste de catégories taxonomiques
#     """
#     Entrez.email = email
#
#     # Rechercher le nom du génome dans la base de données nuccore pour obtenir un identifiant taxonomique (taxid)
#     search_handle = Entrez.esearch(db="nuccore", term=genome_name, retmode="xml")
#     search_results = Entrez.read(search_handle)
#     search_handle.close()
#
#     if not search_results["IdList"]:
#         return f"No results found for genome name: {genome_name}"
#
#     # Utiliser le premier identifiant trouvé pour obtenir des informations détaillées
#     sequence_id = search_results["IdList"][0]
#     summary_handle = Entrez.esummary(db="nuccore", id=sequence_id, retmode="xml")
#     summary_results = Entrez.read(summary_handle)
#     summary_handle.close()
#
#     taxid = summary_results[0]['TaxId']
#
#     # Rechercher l'identifiant taxonomique dans la base de données taxonomy
#     taxonomy_handle = Entrez.efetch(db="taxonomy", id=taxid, retmode="xml")
#     taxon_records = Entrez.read(taxonomy_handle)
#     taxonomy_handle.close()
#
#     # Extraire la taxonomie
#     lineage = taxon_records[0]['Lineage'].split('; ')
#     lineage.append(taxon_records[0]['ScientificName'])
#
#     return lineage

# def tax_search_function(n,email) :
#     try :
#         search = re.sub(r'\.\d+$', '', n)
#         #  tax = get_taxonomy_from_genome_name(search,email)
#         return(tax)
#     except (urllib.error.HTTPError, RuntimeError, IndexError, http.client.IncompleteRead, urllib.error.URLError) as u :
#         print(f'{u} for {data[i][0]}')
#         time.sleep(2)


# email = "any@outlook.com"
#
# tax_search = False
#
# for i in tqdm(range(0, 1, batch_size)):
#     try :
#         batch_labels, batch_strs, batch_tokens = batch_converter([data[i]])
#         batch_inputs = batch_tokens
#         batch_inputs = batch_inputs.to(device)
#         sparse_model = sparse_model.to(device)
#
#         proteins_sizes = torch.Tensor([500, 600, 700, 800, 900, 1000, 876, 300, 100, 100])  # 1-dim tensor of protein sizes in the genome
#         selected_protein = torch.Tensor([1])    # Protein for wich we want the attention scores
#
#         embeddings, attentions = get_embeddings(batch_inputs,i, proteins_sizes, selected_protein)
#         # attentions is a list (len = number of layers) of tensors [1, n_heads, selected_protein_size, genome_length]
#
#
#         if tax_search :
#             tax = tax_search_function(data[i][0], email)
#             all_tax.append((data[i][0],tax))
#
#         all_embeddings.append(embeddings)
#         names.append(data[i][0])
#         processed.append(len(data[i][1]))
#
#     except torch.cuda.OutOfMemoryError as tc : # OOM
#
#         # TO RUN ON CPU IF OOM
#         device2 = 'cpu'
#         sparse_model = sparse_model.to(device2)
#         batch_labels, batch_strs, batch_tokens = batch_converter([data[i]])
#         batch_inputs = batch_tokens
#         batch_inputs = batch_inputs.to(device2)
#         embeddings, attentions = get_embeddings(batch_inputs,i, selected_protein,selected_protein)
#
#         if tax_search :
#             tax = tax_search_function(data[i][0], email)
#             all_tax.append((data[i][0],tax))
#
#         all_embeddings.append(embeddings)
#         names.append(data[i][0])
#
#
# print(f'Fragments processed = {len(all_embeddings)}')
#
# if tax_search :
#     with open(f'{saving_folder}taxonomy.pkl', 'wb') as file :
#         pickle.dump(all_tax, file)
#
# print('saving embeddings...')
# save_emb = torch.stack(all_embeddings, dim=0)
# print(save_emb.shape)
#
# torch.save(save_emb, f"{saving_folder}all_embeddings.pt")
# print(f'saved to {saving_folder}all_embeddings.pt')
#
# save_names = True
# if save_names :
#     with open(f'{saving_folder}proteins_names.pkl', 'wb') as file :
#         pickle.dump(names, file)
