# Assemble proteins to get sequences with POLA+HEL+RNR, and extend the context on both side to a maximum length value (max_value)

fasta_file = '/home/thibaut/pol_hel_rnr/annotated_Keep_ENA_votu_embedding_formatted.fasta'
max_value = 11000

from Bio import SeqIO
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

def load_fasta_as_dic(fasta_path):
    sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        record_str = str(record.seq)
        record.seq = Seq(record_str)
        sequences[record.id] = str(record.seq)

    return sequences

def load_fasta_as_tuples(fasta_path):
    sequences = []
    # Parcourir chaque enregistrement du fichier FASTA
    for record in SeqIO.parse(fasta_path, "fasta"):
        record_str = str(record.seq)
        record.seq = Seq(record_str)
        sequences.append((record.id, str(record.seq)))

    return sequences

sequences = load_fasta_as_dic(fasta_file)
sequences_tuples = load_fasta_as_tuples(fasta_file)

uniq_genoms_id = []
for k, v in sequences.items(): 
    genom_id = k.split('_')[1]
    if genom_id not in uniq_genoms_id : 
        uniq_genoms_id.append(genom_id)

print(f"{len(uniq_genoms_id)} genoms")

hel_rnr_pol = {}
for genom in uniq_genoms_id : 
    genom_seqs = []
    with open(fasta_file, 'r') as annotated_fasta : 
        hel_count = 0
        for line in annotated_fasta : 
            id_x = 1
            if line.startswith(f'>ENA_{genom}') :
                if line.split()[1][0:4] == 'RNR_' : 
                    id_x = 0
                    genom_seqs.append(('RNR', line.split()[0][1:]))
                if line.split()[1][0:5] == 'POLA_' : 
                    id_x = 0
                    genom_seqs.append(('POLA', line.split()[0][1:]))
                if line.split()[1][0:4] == 'HEL_' and hel_count ==0 : 
                    id_x = 0
                    genom_seqs.append(('HEL', line.split()[0][1:]))
                    hel_count +=1
                elif line.split()[1][0:4] == 'HEL_' and hel_count > 0 : 
                    id_x = 0
                    genom_seqs.append((f'HEL{hel_count}', line.split()[0][1:]))
                    hel_count +=1
                elif id_x == 1 : 
                    genom_seqs.append(('X', line.split()[0][1:]))

    hel_rnr_pol[genom] = genom_seqs

# For POLA - RNR - HEL sequences limits

def extract_non_x_sequences(genome_dict, sequence_data, num_before=0, num_after=0 ):
    result = {}
    i = 0
    for genome_id, tuples in genome_dict.items():
        first_non_x_found = False
        last_non_x_index = -1
        indices = {}
        seq = ''
        sequences_running = []
        for index, (key, value) in enumerate(tuples):
            if key != 'X':
                if not first_non_x_found:
                    first_non_x_index = index 
                    first_non_x_found = True
                last_non_x_index = index  
            
            sequences_running.append(sequence_data[value])

        num_before = 0
        num_after = 0

        start_index = first_non_x_index - num_before
        end_index = last_non_x_index + num_after + 1 
        seq_str = ' '.join(sequences_running[start_index:end_index])
        
        a = True

        while len(seq_str) <= max_value and a == True : 
            a = False
            if first_non_x_index - num_before > 0 : 
                a = True
                num_before += 1
            if last_non_x_index + num_after + 1 < len(tuples) : 
                a = True
                num_after +=1 
            
            start_index = first_non_x_index - num_before
            end_index = last_non_x_index + num_after + 1 

            seq_str = ' '.join(sequences_running[start_index:end_index])
        

        selected_tuples = tuples[start_index:end_index]

        for idx, (key, value) in enumerate(selected_tuples, start=start_index):
            if key != 'X':
                indices[key] = idx
        if selected_tuples:
            result[genome_id] = (selected_tuples, tuple((k, indices[k]) for k in indices))
        i+=1

    return result


result = extract_non_x_sequences(hel_rnr_pol, sequences)
print(result['AY954970'])

triples = []
for k, v in result.items() : 
    triples.append(len(v[1]))

def create_fasta_file(result, sequence_data, out_file):
    new_sequences = {}

    with open(out_file, "w") as fasta_file:
        all_length = []
        for genom_id, information in result.items():
            seq = ''
            annotated_ids = ''
            length_ids = ''
            total_length = 0

            for prot in information[0] : 
                new_sequences[prot[1]] = sequence_data[prot[1]]
                seq += sequence_data[prot[1]]
                length_ids += f'_{len(sequence_data[prot[1]])}'
                total_length += len(sequence_data[prot[1]])

            start_prot = int(information[0][0][1].split('_')[-1])-1
            end_prot = int(information[0][-1][1].split('_')[-1])-1

            for annotated_tuples in information[1] : 

                annotated_ids += f'_{annotated_tuples[0]}_{annotated_tuples[1]}'
            
            seq_id = f"{genom_id}_start_{start_prot}_end_{end_prot}{annotated_ids}_prots{length_ids}"

            
            fasta_file.write(f">{seq_id}\n{seq}\n")
            # print(genom_id, total_length)

            all_length.append(total_length)

            

    return all_length, new_sequences



def save_dic_as_fasta(sequences, output_path):
    records = []

    for record_id, sequence in sequences.items():
        seq_obj = Seq(sequence)
        seq_record = SeqRecord(seq_obj, id=record_id, description="")
        records.append(seq_record)

    with open(output_path, 'w') as output_file:
        SeqIO.write(records, output_file, "fasta")


out_file = '/home/thibaut/Keep_assembled_annotated_sequences_corr_annot_11k.fasta'
all_length, new_seqs = create_fasta_file(result, sequences, out_file)
# print(all_length)
out_file2 = '/home/thibaut/Keep_11k_proteins.fasta'
save_dic_as_fasta(new_seqs, out_file2)

print("Assembled sequences informations :")
print(f"{len(all_length)} genoms")
print(f"Max : {max(all_length)}")
print(f"Mean : {sum(all_length)/len(all_length)}")
print(f"Min : {min(all_length)}")

