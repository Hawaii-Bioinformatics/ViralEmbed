from transformers import EsmForSequenceClassification, EsmModel, AutoConfig, EsmConfig, EsmForTokenClassification, EsmTokenizer
from peft import PeftModel
import torch
import esm
from Bio import SeqIO
import matplotlib.pyplot as plt
import os
from Bio.Seq import Seq
from model import SparseForTokenClassification, SparseForMaskedLM
from config_model import SparseConfig
from torch import nn as nn
import re
import itertools
from tqdm import tqdm
from Bio import Entrez
import random 
import urllib
import http
import pickle

import torch
print(torch.version.cuda)  
print(torch.cuda.is_available())

def load_fasta_as_tuples(fasta_path):
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        record_str = str(record.seq)
        record.seq = Seq(record_str)
        sequences.append((record.id, str(record.seq)))
    return(sequences)


model_3b, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
print('alphabet loaded')

num_gpus = torch.cuda.device_count()
print(f"Nombre de GPU disponibles : {num_gpus}")

for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
# device = 'cpu'
print(device)

genom_fasta = '' # genomes fasta file

genoms = load_fasta_as_tuples(genom_fasta)
print(f'initial number of genomes : {len(genoms)}')
genomes = [genome for genome in genoms if len(genome[1])<13000]
data = genomes
print(f'Number of genomes = {len(genomes)}')

import time
time.sleep(20)

saving_folder = '' # Saving folder for embeddings. 

# MODEL
config = SparseConfig()
sparse_model = SparseForTokenClassification(config=config)

checkpoint = torch.load('./pytorch_model.bin', map_location='cpu')
sparse_model.load_state_dict(checkpoint,strict=False)
sparse_model = sparse_model.to(device)
sparse_model = sparse_model.eval()

model_state_dict = checkpoint['model_state_dict']



# EMBEDDINGS 

def get_embeddings(batch_inputs):
    with torch.no_grad():
        outputs = sparse_model(input_ids=batch_inputs, output_attentions = False) 
        embeddings, _ = outputs.logits, outputs.attentions
    return embeddings

# TAXONOMY 
Entrez.email = "" # Email for NCBI. 

def get_taxonomy_info(virus_name):
    search_handle = Entrez.esearch(db="taxonomy", term=virus_name)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    
    if search_results["IdList"]:
        tax_id = search_results["IdList"][0]
        fetch_handle = Entrez.efetch(db="taxonomy", id=tax_id, retmode="xml")
        data = Entrez.read(fetch_handle)
        fetch_handle.close()
        
        taxonomy = data[0]
        tax_dict = {d['Rank']: d['ScientificName'] for d in taxonomy['LineageEx']}
        tax_dict[taxonomy['Rank']] = taxonomy['ScientificName']
        
        return {
            'superkingdom': tax_dict.get('superkingdom', 'N/A'),
            'class': tax_dict.get('class', 'N/A'),
            'order': tax_dict.get('order', 'N/A'),
            'family': tax_dict.get('family', 'N/A'),
            'genus': tax_dict.get('genus', 'N/A'),
            'species': tax_dict.get('species', 'N/A'),
        }
    else:
        return {'superkingdom': 'unclassified',
            'class': 'unclassified',
            'order': 'unclassified',
            'family': 'unclassified',
            'genus': 'unclassified',
            'species': 'unclassified',
        }

names = []
taxes = {}
for genome in tqdm(data) :
    try : 
        
        batch_labels, batch_strs, batch_tokens = batch_converter([genome])
        batch_inputs = batch_tokens
        batch_inputs = batch_inputs.to(device)
        sparse_model = sparse_model.to(device)

        embed = get_embeddings(batch_inputs)
        
        name = genome[0]

        saving_name = name.split('_')[0].replace('/','-')
        tax = get_taxonomy_info(saving_name)

        for k in range(len(name.split('_')[1:])) : 
            taxes[saving_name+f'_{k}'] = tax

        names.append(saving_name)
        torch.save(embed, f'{saving_folder}{saving_name}_genome.pt')
        # print(f'{saving_folder}{name}_genome.pt saved')

    except (urllib.error.HTTPError, RuntimeError, IndexError, http.client.IncompleteRead, urllib.error.URLError, torch.cuda.OutOfMemoryError) as e : 
        print(f'Error for {saving_name}')
        print(e)
        for _ in range(len(name.split('_')[1:])) : 
            taxes[name] = {'superkingdom': 'unclassified', 'class': 'unclassified', 'order': 'unclassified', 'family': 'unclassified', 'genus': 'unclassified', 'species': 'unclassified'}
        

print(embed.shape)

with open(f'{saving_folder}names.pkl', 'wb') as file: 
    pickle.dump(names, file)

with open(f'{saving_folder}taxes.pkl', 'wb') as file2: 
    pickle.dump(taxes, file2)