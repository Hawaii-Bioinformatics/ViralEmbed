# Action Items Summary for GigaScience Data Submission
## Manuscript: GIGA-D-25-00436

**Full Details:** See `answers_data_and_code.md`

---

## ✅ COMPLETED ITEMS

### 1. GitHub Repository Requirements
- [x] **OSI License:** MIT License added (`LICENSE.md`)
- [x] **Data License:** CC0 1.0 for data files (`LICENSE-DATA.md`)
- [x] **Detailed README:** Comprehensive README.md with installation, usage, examples
- [x] **Additional Documentation:** CLAUDE.md, large_context/scripts/README.md, pair_ranking/Readme.md

### 2. Data Files in Repository
- [x] **Training data:** `paper_data/genomes_dataset_train.fa` (30 MB)
- [x] **Protein sequences:** `paper_data/viral.1.protein.faa` (207 MB)
- [x] **Protein metadata:** `paper_data/viral.1.protein.gpff` (1.9 GB)
- [x] **Test genomes:** `paper_data/100_genomes_lv.faa` (294 KB)
- [x] **83 validation genomes (POLA/RNR/HEL):** `paper_data/pola_rnr_hel.pkl` (7.2 KB)
- [x] **PPI scores:** `paper_data/pkl_pairs_gen5.tar.gz` (11 MB, 6,789 genomes)
- [x] **Example data:** `large_context/scripts/data/` (includes realistic use case)
- [x] **Test files:** `test_files/` (attention matrices, embeddings)

### 3. List of Loci
- [x] **83 viral genomes list:** Available in `paper_data/pola_rnr_hel.pkl`
  - Contains POLA, RNR, and HEL protein pairs
  - Used for fine-tuning validation (Figure 2)
  - Verified: 83 genomes in file

---

## ⚠️ URGENT ACTION REQUIRED (Within 3 Days)

### 1. Upload Model Weights to FTP Server

**FTP Credentials:**
- **Server:** files.gigadb.org
- **Username:** user455
- **Password:** SUpJtMdfpOvFC
- **Protocol:** FTP (not SFTP)

**Files to Upload:**

```bash
# From local server locations:
# Source: /home/thibaut/mahdi/models/5B/config_and_model.pth
# Source: /home/thibaut/mahdi/models/3C/config_and_model.pth

# On FTP server, create directory structure:
mkdir 5B
mkdir 3C

# Upload files (use FileZilla or similar):
# Upload config_and_model.pth to 5B/
# Upload config_and_model.pth to 3C/
```

**Expected Size:** ~5 GB total (~2.5 GB each)

### 2. Generate MD5 Checksums

```bash
# After uploading to FTP, compute checksums:
md5sum 5B/config_and_model.pth > md5sums.txt
md5sum 3C/config_and_model.pth >> md5sums.txt

# Upload md5sums.txt to FTP server
```

### 3. Create README.txt for FTP Server

**Content:** (Template provided in `answers_data_and_code.md` Section 7)

Upload `readme.txt` to FTP server root with:
- File descriptions
- MD5 checksums
- GitHub repository link
- Loading instructions
- Citation information

### 4. Notify Editors

Send email to `editorial@gigasciencejournal.com` (cc: `database@gigasciencejournal.com`) with:

```
Subject: Data Upload Complete - GIGA-D-25-00436

Dear Editors,

We have completed the data availability requirements for manuscript GIGA-D-25-00436.

✅ GitHub Repository: https://github.com/Hawaii-Bioinformatics/ViralEmbed
   - All code and most data available
   - OSI-approved license (MIT)
   - CC0 license for data files
   - Detailed documentation

✅ FTP Server Upload Complete:
   - User: user455
   - Files: LV-5B and LV-3C model weights (~5 GB)
   - MD5 checksums provided in md5sums.txt
   - readme.txt included

See attached detailed response document for complete information.

Best regards,
[Your Name]
```

---

## 🔄 MEDIUM PRIORITY (Within 1 Week)

### 1. SciCrunch RRID Registration

**Website:** https://scicrunch.org/

**Registration Information:**
- **Tool Name:** ViralEmbed
- **Description:** Deep learning framework for analyzing viral protein sequences at genomic scale using transformer-based models with sparse attention mechanisms
- **URL:** https://github.com/Hawaii-Bioinformatics/ViralEmbed
- **License:** MIT License
- **Language:** Python
- **Keywords:** protein language model, viral genomics, attention mechanism, protein-protein interaction, deep learning

**After Registration:**
- Add RRID to manuscript text
- Update README.md with RRID

### 2. DOME-ML Annotation

**Website:** https://dome.dsw.elixir-europe.org

**Tutorial:** https://www.youtube.com/watch?v=QNPzQrIeTkk

**Key Information to Provide:**

#### Data (D):
- Training: 14,436 viral genomes (NCBI Virus, June 2024)
- Test: 100 genomes (100 species)
- Validation: 83 genomes (POLA/RNR/HEL)

#### Optimization (O):
- Loss: Masked Language Modeling
- Optimizer: Adam
- Hardware: Multi-GPU (NVIDIA >= 16 GB VRAM)

#### Model (M):
- Architecture: Transformer (33 layers, 20 heads, 1280 hidden)
- Base: ESM2-650M
- Fine-tuning: LongLoRA + ALiBi
- Max length: 61,000 aa
- Parameters: 650M (~2.5 GB)

#### Evaluation (E):
- Metrics: Perplexity, Silhouette, F1, Precision, Recall, AUC
- Tasks: Clustering, Classification, PPI prediction
- Cross-validation: 10-fold

**Publishing Journal:** GigaScience

**After Completion:**
- Download annotation (JSON/text)
- Save copy in repository
- Email copy to editors

---

## 📋 INFORMATION FOR YOUR REFERENCE

### Current Repository Status

**Public URL:** https://github.com/Hawaii-Bioinformatics/ViralEmbed

**Total Data Size:**
- In GitHub: ~2.15 GB
- To upload (FTP): ~5 GB model weights
- **Total:** ~7.15 GB

### Data Files Summary

| File | Location | Size | Description |
|------|----------|------|-------------|
| Training genomes | `paper_data/genomes_dataset_train.fa` | 30 MB | Raw training data |
| Protein sequences | `paper_data/viral.1.protein.faa` | 207 MB | Deduplicated proteins |
| Protein metadata | `paper_data/viral.1.protein.gpff` | 1.9 GB | GenPept format metadata |
| Test genomes | `paper_data/100_genomes_lv.faa` | 294 KB | 100 species for evaluation |
| POLA/RNR/HEL | `paper_data/pola_rnr_hel.pkl` | 7.2 KB | 83 validation genomes |
| PPI scores | `paper_data/pkl_pairs_gen5.tar.gz` | 11 MB | 6,789 genomes |
| Example data | `large_context/scripts/data/` | Various | Working examples |
| Test files | `test_files/` | Various | Attention matrices |

### Train/Test Splits

**Model Training:**
- Training set: `paper_data/genomes_dataset_train.fa`
- Validation: Standard split during training (implicit)

**Classification Evaluation:**
- Test size: 50% (file: `src/classifier/classifier.py`, line 102)
- Random seed: 42
- Cross-validation: 10-fold

**Clustering Evaluation:**
- No split needed (unsupervised)
- Test data: `paper_data/100_genomes_lv.faa`

### Questions for Authors

If any of the following information is unclear, please consult with co-authors:

1. **Training Hyperparameters:**
   - Number of training epochs
   - Batch size used
   - Learning rate schedule
   - Training duration/time

2. **Pfam Annotations:**
   - Are processed Pfam annotation files available?
   - If not, instructions for regeneration are in manuscript (HMMER3 on Pfam 35.0)

3. **Model Versions:**
   - Confirm LV-3C and LV-5B are the exact models used in manuscript
   - Confirm file paths on server:
     - `/home/thibaut/mahdi/models/5B/config_and_model.pth`
     - `/home/thibaut/mahdi/models/3C/config_and_model.pth`

---

## 🎯 CHECKLIST

### Immediate (3 days):
- [ ] Upload LV-5B model to FTP (5B/config_and_model.pth)
- [ ] Upload LV-3C model to FTP (3C/config_and_model.pth)
- [ ] Generate MD5 checksums
- [ ] Create and upload readme.txt to FTP
- [ ] Email editors with confirmation

### Short-term (1 week):
- [ ] Register at SciCrunch.org for RRID
- [ ] Complete DOME-ML annotation
- [ ] Submit DOME-ML to registry
- [ ] Save DOME annotation copy
- [ ] Update manuscript with RRID

### Follow-up:
- [ ] Respond to any editor questions
- [ ] Verify FTP access works for reviewers
- [ ] Monitor for reviewer feedback on data availability

---

## 📧 Contact for Questions

**GigaScience Data Curator:**
- Email: database@gigasciencejournal.com

**Editorial Office:**
- Email: editorial@gigasciencejournal.com

**Primary Author:**
- Dr. Mahdi Belcaid: mahdi@hawaii.edu

**Co-author (Technical):**
- Thibaut Dejean: thib.dejean69@gmail.com

---

## 📎 Related Files

- **Detailed Answers:** `answers_data_and_code.md` (659 lines, comprehensive)
- **Main README:** `README.md`
- **Technical Docs:** `CLAUDE.md`
- **Licenses:** `LICENSE.md`, `LICENSE-DATA.md`

---

**Document Created:** October 30, 2025
**Editor Deadline:** 3 days from receipt (October 22 email)
**Priority:** URGENT
