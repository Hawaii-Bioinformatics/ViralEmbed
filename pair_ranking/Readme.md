Script to rank annotated + clustered pairs

# For mmseqs clustering : 
cluster_fasta_annot.py + annot_tsv.py : Complete mmseqs clustering with X/Y clusters, retaining POL/HEL/RNR  
Original fasta + mmseqs tsv output -> fully annotated fasta -> annotated tsv

# For complete clustering : 
todo.py : Annotate original fasta file with clusters from using tsv file  
Clustering tsv output -> annotated fasta

# For pair ranking : 
get_annotations.py : Create .txt files (1 per fragment) with annotated proteins (used for pair ranking labels)  
Annotated fasta -> folder  
  
pair_ranking_script.py : Complete script for pair ranking, using attentions and cluster annotations (can plot/save figures)  
Save rankings in a .pkl file  
attentions + annotated fasta file + folder -> pair ranks, node graph, heatmap  
