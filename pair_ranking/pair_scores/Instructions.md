# assemble_fasta_length.py
Assemble proteins to get sequences with POLA+HEL+RNR, and extend the context on both side to a maximum length value (max_value). 
Create a dataset fasta file (1)

# get_annotations.py
Create .txt files for each genome with all the proteins in the fasta file (1) and their annotations. 
Format : protein_id \t Annotation
Create a fasta file for every proteins in the assembled fasta file (1)

# pair_ranking_no_annot.py 
Create .pkl for every genome with all the scores of every pair of proteins. 


