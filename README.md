# ViralEmbed

**Extending Protein Language Models to Viral Genomic Scale**

ViralEmbed is a deep learning framework for analyzing viral protein sequences at genomic scale using transformer-based models with sparse attention mechanisms. The tool processes complete viral genomes (up to 61,000 amino acids) to predict protein-protein interactions and identify functional relationships between viral proteins through attention-based contact analysis.

## Features

- Long-context processing: Handle viral genomes up to 61,000 tokens using block-wise sparse attention using:
  - Protein-protein interaction prediction: Compute interaction scores between all protein pairs in a genome
  - Attention-based contact analysis: Extract residue-level contacts from transformer attention patterns
- Multi-GPU support: Automatic distribution of model layers across available GPUs
- Flexible inference modes: Extract embeddings, full attention matrices, or specific protein pair interactions

## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Input/Output Formats](#inputoutput-formats)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Hawaii-Bioinformatics/ViralEmbed.git
cd ViralEmbed
```

### 2. Set up Python environment

ViralEmbed requires Python 3.9 or higher. We recommend using conda:

```bash
conda create -n viralembed python=3.9
conda activate viralembed
```

### 3. Install dependencies

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio

# Install ESM (Facebook's protein language model)
pip install fair-esm

# Install other dependencies
pip install transformers biopython scipy matplotlib seaborn networkx
```

### 4. Download model checkpoint

**Note**: Model checkpoints are not included in the repository.

Please contact the authors to obtain trained model weights.

Place the checkpoint file in an appropriate directory (e.g., `./models/3C/config_and_model.pth`).

## Dependencies

### Required

- **Python**: ≥ 3.9
- **PyTorch**: ≥ 1.8
- **fair-esm**: For ESM2-650M tokenizer
- **transformers**: HuggingFace utilities
- **Biopython**: FASTA file parsing

### Optional (for analysis and visualization)

- **scipy**: Sparse matrix operations
- **matplotlib**: Plotting
- **seaborn**: Advanced visualizations
- **networkx**: Network graph analysis
- **memory_profiler**: Memory profiling

## Quick Start

```python
import torch
import esm
from large_context.scripts.large_prot_encoding.model import SparseForTokenClassification

# Load ESM2 tokenizer
_, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

# Load ViralEmbed model
checkpoint = torch.load('./models/3C/config_and_model.pth',
                       map_location='cpu', weights_only=False)
model = SparseForTokenClassification(config=checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to('cuda')
model.eval()

# Process a genome with multiple proteins
genome = [('genome_1', 'MLKKLSVFLIMLSVFLILKKLSVFLI')]  # Concatenated proteins
protein_lengths = [10, 7, 9]  # Length of each protein

# Tokenize
batch_labels, batch_strs, batch_tokens = batch_converter(genome)

# Get embeddings and attentions
with torch.no_grad():
    outputs = model(
        input_ids=batch_tokens,
        proteins_sizes=torch.tensor(protein_lengths),
        output_attentions=True,
        two_step_selection=True  # Returns ranked protein pairs
    )

# outputs.logits: Protein embeddings [1, seq_len+2, 1280]
# outputs.attentions: Ranked protein pairs dictionary {(i,j): score}
```

## Usage Examples

### Example 1: Extract Embeddings for a Single Protein

```python
# Single protein sequence
data = [('protein_1', 'MLKKLSVFLI')]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# Get embeddings
with torch.no_grad():
    outputs = model(input_ids=batch_tokens, output_attentions=True)

# Extract sequence embeddings (excluding CLS and EOS tokens)
embeddings = outputs.logits[0, 1:-1, :]  # Shape: [10, 1280]
```

### Example 2: Process a Complete Viral Genome

```python
from Bio import SeqIO

# Load genome FASTA with header encoding protein boundaries
# Header format: >genome_id_prots_500_600_700
def load_genome_fasta(fasta_path):
    records = list(SeqIO.parse(fasta_path, "fasta"))
    header = records[0].id
    sequence = str(records[0].seq)

    # Extract protein sizes from header
    sizes = [int(x) for x in header.split('_prots_')[1].split('_')]
    return [(header, sequence)], sizes

genome, protein_sizes = load_genome_fasta('viral_genome.fa')
batch_labels, batch_strs, batch_tokens = batch_converter(genome)

# Run inference
with torch.no_grad():
    outputs = model(
        input_ids=batch_tokens,
        proteins_sizes=torch.tensor(protein_sizes),
        output_attentions=True
    )

# outputs.attentions: List of attention matrices, one per layer
# Each matrix has shape [1, n_heads, seq_len, seq_len]
```

### Example 3: Rank Protein-Protein Interactions

```python
from large_context.scripts.large_prot_encoding.two_step_attention import rank_assembled_pairs

# Get attention matrix (averaged across heads and layers)
attentions = outputs.attentions
avg_attention = torch.stack([att.mean(dim=1) for att in attentions]).mean(dim=0)

# Rank all protein pairs
all_pairs_ranked, ranked_dict = rank_assembled_pairs(
    protein_sizes,
    avg_attention
)

# Display top interactions
print("Top 5 protein-protein interactions:")
for (prot_i, prot_j), score in all_pairs_ranked[:5]:
    print(f"Protein {prot_i} <-> Protein {prot_j}: {score:.4f}")
```

### Example 4: Complete Analysis Pipeline

See the Jupyter notebooks for complete workflows:

- `large_context/scripts/large_prot_encoding/examples.ipynb`: Basic usage examples

  - Loading models and tokenizers
  - Processing single proteins and complete genomes
  - Combining attention heads and layers
  - Ranking protein pairs

- `attention_contact_pipeline.ipynb`: End-to-end contact analysis

  - From FASTA files to contact predictions
  - Visualization of attention patterns

- `pair_ranking/pair_ranking_script.py`: Complete pair ranking pipeline
  - Clustering and annotation
  - Batch processing of viral genomes
  - Generating heatmaps and network graphs

## Input/Output Formats

### Input Formats

#### 1. FASTA Files (Genome Assembly)

For model inference, provide viral genomes as concatenated protein sequences:

```
>NC_001234_prots_500_600_700_350
MLKKLSVFLI...QWERTYASDF...CVBNMASDFG...TYUIOPHJKL
```

- **Header format**: `>{genome_id}_prots_{size1}_{size2}_{size3}...`
- **Sequence**: Concatenated amino acid sequences of all proteins in the genome
- The model uses protein sizes to identify individual proteins within the concatenated sequence

#### 2. FASTA Files (Annotated Proteins)

For pair ranking with labels:

```
>protein_0 [cluster_1] POLA
MLKKLSVFLI...
>protein_1 [cluster_2] HEL
QWERTYASDF...
```

- Used by annotation pipeline (`pair_ranking/get_annotations.py`)
- Annotations: POLA (polymerase), HEL (helicase), RNR (ribonucleotide reductase), etc.

#### 3. Annotation Files (.txt)

Tab-separated protein labels (one per genome):

```
protein_0	POLA
protein_1	HEL
protein_2	cluster_1
protein_3	RNR
```

### Output Formats

#### 1. Embeddings

- **Format**: PyTorch tensor `[1, seq_len+2, 1280]`
- **Content**: Per-residue embeddings (includes CLS and EOS tokens)
- **Usage**: Downstream machine learning tasks, clustering, visualization

#### 2. Attention Matrices (.pt)

- **Raw output**: List of tensors `[1, n_heads, seq_len, seq_len]` (one per layer)
- **Aggregated**: Single tensor `[1, seq_len, seq_len]` (averaged across heads/layers)
- **File naming**: `{genome_id}_full_attentions_weighted.pt`

#### 3. Ranked Protein Pairs (.pkl)

Dictionary format saved with pickle:

```python
{
    (0, 1): 0.234,  # Protein 0 <-> Protein 1
    (0, 2): 0.189,  # Protein 0 <-> Protein 2
    (1, 2): 0.156,  # Protein 1 <-> Protein 2
    ...
}
```

Scores are APC-normalized attention values (higher = stronger interaction).

#### 4. Visualizations

- **Heatmaps**: Protein-protein interaction matrices (matplotlib/seaborn)
- **Network graphs**: Protein interaction networks (NetworkX)

## Documentation

Detailed documentation is available in:

- **[Architecture.md](Architecture.md)**: Complete architecture guide and implementation details
- **[large_context/scripts/README.md](large_context/scripts/README.md)**: Pair ranking script documentation
- **[pair_ranking/Readme.md](pair_ranking/Readme.md)**: Annotation and clustering pipeline

### Key Components

- **Model Architecture**: `large_context/scripts/large_prot_encoding/model.py`
- **Configuration**: `large_context/scripts/large_prot_encoding/config_model.py`
- **Inference**: `large_context/scripts/large_prot_encoding/inference.py`
- **Pair Ranking**: `large_context/scripts/large_prot_encoding/two_step_attention.py`
- **Block Attention**: `large_context/scripts/large_prot_encoding/bio_attention.py`

## Model Architecture

The projects extends ViralEmbed extends ESM2 architecture with:

- 33 transformer layers with 20 attention heads each
- Hidden dimension: 1280
- Max sequence length: 61,000 amino acids
- Rotary position embeddings (RoPE) for long-range dependencies
- Block-wise sparse attention for sequences > 12,000 tokens
- APC normalization for interaction scores

## System Requirements

### Recommended Hardware

- GPU: NVIDIA GPU with ≥ 16 GB VRAM (V100, A100, or RTX 3090/4090)
- RAM\*\*: ≥ 32 GB for processing long genomes
- Storage: ≥ 10 GB for model checkpoint and data

### Multi-GPU Support

The model automatically distributes layers across multiple GPUs if available:

```python
# Automatically uses all available GPUs
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPUs")
```

## Citation

If you use ViralEmbed in your research, please cite:

```bibtex
@article{viralembed2024,
  title={ViralEmbed: Extending Protein Language Models to Viral Genomic Scale},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024},
  url={https://github.com/Hawaii-Bioinformatics/ViralEmbed}
}
```

_Publication details will be updated upon acceptance._

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## Support

For questions, issues, or feature requests:

- Email:
  - Thibaut Dejean thib.dejean69@gmail.com
  - Mahdi Belcaid mahdi@hawaii.edu

## License

MIT

## Acknowledgments

- ESM2 model from Meta AI (facebook research)

* Last updated: 10/22/2025
