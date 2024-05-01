import csv

input_tsv_path = '/home/thibaut/mmseqs_clustering/out_cluster_cluster.tsv' # mmseqs clustering output
input_fasta_path = '/home/thibaut/pol_hel_rnr/annotated_Keep_ENA_votu_embedding_formatted.fasta' # Fasta file with sequences and description 
output_fasta_path = '/home/thibaut/pol_hel_rnr/annotated_Keep_ENA_votu_embedding_formatted_clustered_all.fasta'

# Reading clusters from the TSV file
clusters = {}
with open(input_tsv_path, 'r') as file:
    tsv_reader = csv.reader(file, delimiter='\t')
    for row in tsv_reader:
        if row:
            cluster_id = row[0].split('_')[1]
            for protein_id in row:
                clusters[protein_id] = cluster_id

# Write FASTA file
cluster_counter = {}
with open(input_fasta_path, 'r') as fasta_file, open(output_fasta_path, 'w') as output_file:
    for line in fasta_file:
        if line.startswith('>'):
            protein_id = line.split(' ')[0][1:]  
            description = ' '.join(line.split(' ')[1:]).strip()
            if description in ["Uncharacterized protein", "Phage protein"]:
                cluster = clusters.get(protein_id, 'Unknown')
                if cluster not in cluster_counter:
                    cluster_counter[cluster] = 1
                else:
                    cluster_counter[cluster] += 1
                description = f"X_cluster_{cluster_counter[cluster]}"

            elif description[0:3] not in ["POL", "RNR", "HEL"] : 
                cluster = clusters.get(protein_id, 'Unknown')
                if cluster not in cluster_counter:
                    cluster_counter[cluster] = 1
                else:
                    cluster_counter[cluster] += 1
                description = f"Y_cluster_{cluster_counter[cluster]}"
                
            output_file.write(f'>{protein_id} {description}\n')
        else:
            output_file.write(line)

print("Fasta file generated")


# Set to store unique cluster identifiers
unique_clusters = set()

# Read TSV file
with open(input_tsv_path, 'r') as file:
    tsv_reader = csv.reader(file, delimiter='\t')
    for row in tsv_reader:
        for protein_id in row:
            cluster_id = protein_id.split('_')[1]
            unique_clusters.add(cluster_id)

# Number of unique clusters
print(f'Number of unique clusters : {len(unique_clusters)}')
