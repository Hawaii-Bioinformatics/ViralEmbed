import os

def extract_protein_annotations(fasta_file, output_dir):
    # Create output directory if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    annotations = {}
    with open(fasta_file, 'r') as file:
        for line in file:
            if line.startswith('>'):
                parts = line.split()
                genome_id = parts[0].split('_')[1]
                protein_annotation = ' '.join(parts[1:])
                
                if genome_id not in annotations:
                    annotations[genome_id] = []
                annotations[genome_id].append((parts[0][1:], protein_annotation))
    
    # Write annotations in separate files
    for genome_id, annotation_list in annotations.items():
        with open(os.path.join(output_dir, f"{genome_id}.txt"), 'w') as out_file:
            for annotation in annotation_list:
                out_file.write(annotation[0] +'\t' + annotation[1] + '\n')


def filter_annotations(input_dir, fasta):
    # Filter the sequences between first and last ['HEL', 'POL', 'RNR']
    files = os.listdir(input_dir)
    
    for filename in files:
        a = False
        if filename.endswith(".txt"):
            path = os.path.join(input_dir, filename)
            with open(path, 'r') as file:
                lines = file.readlines()
            
            # Identify the first and last valid IDs
            first_id_index = None
            last_id_index = None
            
            genom_id = filename[:-4]

            with open(fasta, 'r') as ff : 
                for fline in ff : 
                    if fline.startswith(f'>{genom_id}'):
                        a = True 
                        first_id_index = int(fline.split('_')[2])
                        last_id_index = int(fline.split('_')[4])
                    
            # Write filtered lines to a new file or rewrite on the old one
            if first_id_index is not None and last_id_index is not None:
                with open(path, 'w') as file:
                    file.writelines(lines[first_id_index:last_id_index+1])

        if a == False :
            print(genom_id)
            print('not found')

fasta_file = '/home/thibaut/pol_hel_rnr/annotated_Keep_ENA_votu_embedding_formatted.fasta'  
output_dir = '/home/thibaut/KEEP_80_v3/'

fasta2 = '/home/thibaut/Keep_assembled_annotated_sequences_corr_annot_11k.fasta'
extract_protein_annotations(fasta_file, output_dir)
filter_annotations(output_dir, fasta2)
