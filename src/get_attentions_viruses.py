from transformers import EsmForSequenceClassification, EsmModel, AutoConfig, EsmConfig, EsmForTokenClassification
import torch
import os 
from tqdm import tqdm
import logging
from memory_profiler import profile
import psutil

logging.getLogger("transformers").setLevel(logging.ERROR) 

def parse_fasta(file_path):
    with open(file_path, 'r') as file:
        genomes = {}
        
        current_genome = None
        current_sequence = []
        
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_genome and current_sequence:
                    length = len(''.join(current_sequence))
                    genomes[current_genome].append((current_id, length))
                
                parts = line[1:].split('_')
                genome_id = line[1:6]
                sequence_id = int(parts[-1])
                
                current_genome = genome_id
                current_id = sequence_id
                current_sequence = []
                
                if genome_id not in genomes:
                    genomes[genome_id] = []
            else:
                # Accumuler la séquence
                current_sequence.append(line)
        
        # Ne pas oublier d'ajouter la dernière séquence
        if current_genome and current_sequence:
            length = len(''.join(current_sequence))
            genomes[current_genome].append((current_id, length))
    
    # Transformer le dictionnaire en liste de tuples comme demandé
    result = []
    for genome_id, sequences in genomes.items():
        result.append((genome_id, *sequences))
    
    return result

"""
file_path = '/home/thibaut/String/Influenza/Influenza_A_proteins.fasta'
parsed_data = parse_fasta(file_path)
"""
#print(parsed_data)

def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    rss_memory = memory_info.rss / (1024 * 1024)  # Convertir de bytes à megabytes
    mem_usage = process.memory_percent('rss')
    print(f"Current memory usage: {rss_memory:.2f} MB, {mem_usage:.2f}%")
    

@profile
def apc(x):
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)
    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    x.sub_(avg) # in-place to reduce memory
    del avg
    return x

def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    x_t = x.clone().transpose(-1, -2)
    x += x_t
    return x

@profile
def process_sequences(parsed_data, base_embeddings_folder, weights):
    for index, (genome_id, *sequence_info) in tqdm(enumerate(parsed_data)):
        sequence_dir = f"{base_embeddings_folder}/sequence_{index}_mutual_information_apc"
        os.makedirs(sequence_dir, exist_ok=True)
        tensor_name = f"virus_{index}_full_attentions.pt"
        tensor_path = os.path.join(base_embeddings_folder, tensor_name)
        contact_tensor = torch.load(tensor_path) 
        contact_tensor_reduced = contact_tensor[:,:,1:-1, 1:-1]
        del contact_tensor
        #tensor_apc = apc(symmetrize(contact_tensor_reduced))
        #del contact_tensor_reduced
        weights = weights.squeeze()
        weighted_attentions = contact_tensor_reduced * weights.view(660, 1, 1)
        sum_attention = weighted_attentions.sum(dim=1, keepdim=True)
        #del tensor_apc
        sum_attention = sum_attention.squeeze(1)
        liste_cumulee = []
        somme_cumulee = 0

        for id, valeur in sequence_info:
            somme_cumulee += valeur
            liste_cumulee.append((id, somme_cumulee))

        limite=sum_attention.size(1)
        liste_tronquee = [tup for tup in liste_cumulee if tup[1] <= limite]
        sequence_info = sequence_info[:len(liste_tronquee)]

        for i in range(len(sequence_info)):
            for j in range(i + 1, len(sequence_info)):
                prot1, len1 = sequence_info[i]
                prot2, len2 = sequence_info[j]
                end1 = liste_cumulee[i][1]
                end2 = liste_cumulee[j][1]

                block1 = sum_attention[0][end1-len1: end1, end2-len2: end2]
                block2 = sum_attention[0][end2-len2: end2, end1-len1: end1]
                
                block2_transposed = block2.t()
                mutual_information_tensor = block1 + block2_transposed
                mutual_apc = apc((mutual_information_tensor))
            
                torch.save(mutual_apc, f"{sequence_dir}/{prot1}_{prot2}_attention.pt")

"""
base_model_path = "facebook/esm2_t33_650M_UR50D"
model = EsmForTokenClassification.from_pretrained(base_model_path)
for name, param in model.named_parameters():
    if "contact_head" in name : 
        weights = param
        break

# Usage
base_folder = "/home/thibaut/contacts/influenza/"
process_sequences(parsed_data, base_folder, weights)

virus_id = '162145'
proteins_output = f"/home/thibaut/String/{virus_id}/proteins.fasta"
parsed_data = parse_fasta(proteins_output)
base_embeddings_folder = f"/home/thibaut/String/{virus_id}/embeddings/"
process_sequences(parsed_data, base_embeddings_folder, weights)
"""