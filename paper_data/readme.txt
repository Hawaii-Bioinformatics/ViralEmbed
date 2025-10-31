ViralEmbed: Pretrained Model Weights and Data
Manuscript: GIGA-D-25-00436  
Title: Extending Protein Language Models to a Viral Genomic Scale Using Biologically Induced Sparse Attention


Files in This Directory

Model Weights

3C/config_and_model.pth  
LV-3C model checkpoint: 33 layers, 20 attention heads, 1280 hidden dimension, max sequence length 61,000 amino acids. Alternative model variant with different attention patterns.

5B/config_and_model.pth  
LV-5B model checkpoint: Same architecture as LV-3C. Main model for processing viral genomes.

Data Files

100_genomes_lv.faa  
Test dataset containing 100 distinct viral genomes representing 100 species. 

data.tar.gz  
Example genome data from large_context/scripts/data/. Contains example viral genome FASTA files, protein annotations, pre-computed attention tensors, and complete worked examples with notebooks.

genomes_dataset_train.fa  
Training dataset containing concatenated protein sequences from viral genomes. Used for training LV-3C and LV-5B models with masked language modeling.

pkl_pairs_gen5.tar.gz  
Comprehensive protein-protein interaction scores for 6,789 viral genomes. Contains individual pickle files with APC-normalized attention values mapping protein pairs to interaction scores. Used for evaluating attention-based PPI predictions (Section 3.5).

pola_rnr_hel.pkl  
Validation dataset containing 83 viral genome fragments with POLA, RNR, and HEL replication proteins. Used to validate fine-tuning strategy and attention-based PPI detection (Figure 2).

test_files.tar.gz  
Pre-computed attention matrices and embeddings for test genomes. Contains PyTorch tensor files with full attention matrices, protein embeddings, and reference sequences for validation and examples.

viral.1.protein.faa  
Deduplicated viral protein sequences from NCBI Virus database. Complete training sequences in FASTA format.

viral.1.protein.gpff  
Complete protein metadata in GenPept format. Contains genome IDs, species names, taxonomy lineage, and family information.

md5sums.txt  
MD5 checksums for file integrity verification.


Loading the Models

Complete loading instructions, usage examples, and worked notebooks are available in the GitHub repository:

Repository: https://github.com/Hawaii-Bioinformatics/ViralEmbed

Documentation:
- Main README: Installation and quick start guide
- Large context inference: large_context/scripts/README.md
- Pair ranking pipeline: pair_ranking/Readme.md
- Worked example: large_context/scripts/data/realistic_use_case/ (includes Jupyter notebook)
- Complete API documentation: CLAUDE.md

---

System Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.8
- GPU with ≥ 16 GB VRAM (V100, A100, or RTX 3090/4090 recommended)
- RAM ≥ 32 GB

---

License

- Software: MIT License (OSI-approved)
- Data: CC0 1.0 Universal (Public Domain Dedication)

See LICENSE.md and LICENSE-DATA.md in the GitHub repository.

---

Contact

Corresponding Author:  
Mahdi Belcaid  
mahdi@hawaii.edu  
University of Hawaii

Primary-author:  
Thibaut Dejean  
thib.dejean69@gmail.com

RRID: SCR_027596

---

Last updated: October 30, 2025

