def read_interaction_pairs(file_path):
    pairs = set()
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            protein1 = parts[1].split('_')[0]
            protein2 = parts[3].split('_')[0]
            pairs.add((protein1, protein2))
    return pairs

def read_fasta_positions(fasta_file):
    positions = {}
    with open(fasta_file, 'r') as file:
        for line in file:
            if line.startswith('>'):
                parts = line.strip().split('.')
                protein_id = parts[1].split('_')[0]
                position = int(parts[1].split('_')[-1])
                positions[protein_id] = position
    return positions

def calculate_distance(pos1, pos2):
    return abs(pos1 - pos2)

def generate_distance_file(fasta_file, interactions_file, output_file):
    positions = read_fasta_positions(fasta_file)
    relevant_pairs = read_interaction_pairs(interactions_file)
    
    with open(output_file, 'w') as file:
        for protein1, protein2 in relevant_pairs:
            if protein1 in positions and protein2 in positions:
                distance = calculate_distance(positions[protein1], positions[protein2])
                line = f"11320\t{protein1}_I34A1\t11320\t{protein2}_I34A1\t{distance}\n"
                file.write(line)

# Example usage
#input_fasta_file = '/home/thibaut/String/Influenza/Influenza_A_proteins.fasta'
#attention_score_file = '/home/thibaut/contacts/influenza/results.txt'
#output_file = '/home/thibaut/String/Influenza/Influenza_A_distances.txt'
#generate_distance_file(input_fasta_file, attention_score_file ,output_file)


def read_fasta_length(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    protein_lengths = {}
    current_protein = None
    sequence = ""

    for line in lines:
        if line.startswith('>'):
            if current_protein:
                protein_lengths[current_protein] = len(sequence)
            header_parts = line[1:].split('.')  # Remove '>' and split by '.'
            current_protein = header_parts[1].split('_')[0]+'_'+header_parts[1].split('_')[1] # Take second part and split by '_'
            print(current_protein)
            sequence = ""
        else:
            sequence += line.strip()

    # Add the last protein's length
    if current_protein:
        protein_lengths[current_protein] = len(sequence)

    return protein_lengths


def generate_output_file(input_file_path, protein_lengths, output_file_path):
    # print(protein_lengths)
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            protein1 = parts[1]
            protein2 = parts[3]
            length_sum = protein_lengths[protein1] + protein_lengths[protein2]
            outfile.write(f"{parts[0]}\t{parts[1]}\t{parts[2]}\t{parts[3]}\t{length_sum}\n")

#output_file_sizes = '/home/thibaut/String/Influenza_A_sizes.txt'

#protein_lengths = read_fasta_length(input_fasta_file)
#generate_output_file(attention_score_file, protein_lengths, output_file_sizes)