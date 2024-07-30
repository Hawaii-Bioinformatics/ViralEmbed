### Pair-ranking script
===================

This script is used to rank pairs of proteins in a genome 

inputs are
Fasta file of proteins  
Attention file of complete genome.



get_pairs_dataset(fasta, attention_file, labels_file, plot_fragment_heatmap = False)
# get_pairs_dataset("./data/LR699048.fa", "./data/LR699048.pt", "./data/LR699048.txt", plot_fragment_heatmap = False)
# generates the folowing
all_pairs: attention between all pairs of proteins 
all_nc_pairs:  attention between all pairs of non-consecutive proteins
global_length: total number of all pairwise comparisons ()
non_consecutive_length: total number of all non-consecutive pairwise comparisons
database_dic: same as all_pairs but in dictionary format where key is genome name
all_top_pairs_labeled: same as all_pairs but with protein labels instead of protein indexes. Uses the labels_file to get the labels
all_top_pairs_labeled_nc: same as all_nc_pairs but with protein labels instead of protein indexes. Uses the labels_file to get the labels

to generate attention_file (./data/LR699048.pt), we use the script [large_prot_encoding/inference.py](inference.py) with the following command

data = load_fasta_as_tuples("../data/LR699048.fa")
batch_labels, batch_strs, batch_tokens = batch_converter([data[0]])
embeds = get_embeddings(batch_tokens, [0])

Embeds contains:
  - The embeddings: 1, len(genome) +2 , 1280
  - The attention matrices for each layer. torch.Size([1, 20, 4515, 4515])
    - Each layer is torch.Size([1, 20, len(genome), len(genome)])
