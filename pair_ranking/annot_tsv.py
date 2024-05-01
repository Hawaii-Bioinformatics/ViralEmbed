import csv

# Add clusters ids to .tsv file from Fasta file. 

input_tsv_path = '/home/thibaut/mmseqs_clustering/out_cluster_cluster.tsv'
input_fasta_path = '/home/thibaut/pol_hel_rnr/annotated_Keep_ENA_votu_embedding_formatted_clustered_all.fasta'  
output_tsv_path = '/home/thibaut/mmseqs_clustering/out_cluster_named_ednc.tsv'

protein_descriptions = {}

# Read the FASTA file for protein descriptions
with open(input_fasta_path, 'r') as fasta_file:
    for line in fasta_file:
        if line.startswith('>'):
            parts = line.strip().split(' ')
            protein_id = parts[0][1:]  
            description = ' '.join(parts[1:])  
            protein_descriptions[protein_id] = description

# Read TSV file and write new TSV file
with open(input_tsv_path, 'r') as tsv_file, open(output_tsv_path, 'w', newline='') as out_file:
    tsv_reader = csv.reader(tsv_file, delimiter='\t')
    tsv_writer = csv.writer(out_file, delimiter='\t')
    
    for row in tsv_reader:
        if row:
            protein1_id = row[0]
            protein2_id = row[1]
            # Retrieve descriptions for each identifier
            description1 = protein_descriptions.get(protein1_id, 'Unknown')
            description2 = protein_descriptions.get(protein2_id, 'Unknown')
            # Write the new line with the added descriptions
            tsv_writer.writerow([protein1_id, protein2_id, description1, description2])

print("Fichier TSV modifié généré avec succès.")
