### Pair-ranking script
===================

** Please ask us for the necessary models! ** 

This script is used to rank pairs of proteins in a genome 

inputs are: 
- Fasta file of proteins  
- Attention file of complete genome.



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

#### Generating Attention File
to generate attention_file (./data/LR699048.pt), we use the script [large_prot_encoding/inference.py](inference.py) with the following command

data = load_fasta_as_tuples("../data/LR699048.fa")
batch_labels, batch_strs, batch_tokens = batch_converter([data[0]])
protein_sizes = proteins_sizes_from_header(data[0][0])
embeds = get_embeddings(batch_tokens, [0], torch.tensor(protein_sizes), None, pairwise_scores=True)
Above, None, mean we are not selecting a specific pair of proteins to return
pairwise_scores returns all the pairwise scores

Embeds contains:
  - The embeddings: 1, len(genome) +2 , 1280
    - The attention matrices for each layer. torch.Size([1, 20, 4515, 4515])
      - Each layer is torch.Size([1, 20, len(genome), len(genome)])
  - The pairwise scores as a dict. Key is pair and value is attention score non-normalized.  


    embeds = get_embeddings(batch_tokens, [0], torch.tensor(protein_sizes), torch.tensor([13,14]), pairwise_scores=False)
Above returns embeds[0] as The embeddings: 1, len(genome) +2 , 1280
embeds[1] is a list of


