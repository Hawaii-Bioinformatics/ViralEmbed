# FTP Upload Instructions for GigaScience

## Quick Reference

**FTP Server:** files.gigadb.org
**Username:** user455
**Password:** SUpJtMdfpOvFC
**Protocol:** FTP (NOT SFTP)
**Deadline:** Within 3 days of October 22, 2025

---

## Files to Upload

### Model Weights (from server at /home/thibaut/mahdi/models/)

1. **LV-5B Model**
   - Source: `/home/thibaut/mahdi/models/5B/config_and_model.pth`
   - Destination on FTP: `5B/config_and_model.pth`
   - Size: ~2.5 GB

2. **LV-3C Model**
   - Source: `/home/thibaut/mahdi/models/3C/config_and_model.pth`
   - Destination on FTP: `3C/config_and_model.pth`
   - Size: ~2.5 GB

**Total Upload Size:** ~5 GB

---

## Method 1: Using FileZilla (Recommended)

### Step 1: Install FileZilla
```bash
# Ubuntu/Debian
sudo apt-get install filezilla

# macOS
brew install --cask filezilla

# Or download from: https://filezilla-project.org/
```

### Step 2: Configure Connection
1. Open FileZilla
2. File → Site Manager → New Site
3. Enter details:
   - **Protocol:** FTP - File Transfer Protocol
   - **Host:** files.gigadb.org
   - **Port:** 21 (default)
   - **Encryption:** Use explicit FTP over TLS if available
   - **Logon Type:** Normal
   - **User:** user455
   - **Password:** SUpJtMdfpOvFC
4. Click "Connect"

### Step 3: Create Directory Structure
Once connected, in the remote site panel (right side):
1. Right-click → Create directory → Name it "5B"
2. Right-click → Create directory → Name it "3C"

### Step 4: Upload Files
1. Navigate to `/home/thibaut/mahdi/models/5B/` on local side (left panel)
2. Select `config_and_model.pth`
3. Drag to the `5B/` folder on remote side (right panel)
4. Repeat for 3C model
5. Wait for upload to complete (will take time due to size)

---

## Method 2: Using Command Line FTP

### Option A: Standard FTP Client

```bash
# Connect to FTP server
ftp files.gigadb.org

# When prompted, enter:
# Name: user455
# Password: SUpJtMdfpOvFC

# Create directories
ftp> mkdir 5B
ftp> mkdir 3C

# Upload LV-5B model
ftp> cd 5B
ftp> binary
ftp> put /home/thibaut/mahdi/models/5B/config_and_model.pth
# Wait for upload...

# Upload LV-3C model
ftp> cd ../3C
ftp> put /home/thibaut/mahdi/models/3C/config_and_model.pth
# Wait for upload...

# Exit
ftp> bye
```

### Option B: Using lftp (Better for Large Files)

```bash
# Install lftp if needed
sudo apt-get install lftp  # Ubuntu/Debian
brew install lftp           # macOS

# Create upload script
cat > upload_models.sh << 'EOF'
#!/bin/bash

lftp -c "
open -u user455,SUpJtMdfpOvFC files.gigadb.org
mkdir -p 5B
mkdir -p 3C
cd 5B
put /home/thibaut/mahdi/models/5B/config_and_model.pth
cd ../3C
put /home/thibaut/mahdi/models/3C/config_and_model.pth
bye
"
EOF

chmod +x upload_models.sh
./upload_models.sh
```

### Option C: Using ncftp (Alternative)

```bash
# Install ncftp
sudo apt-get install ncftp  # Ubuntu/Debian
brew install ncftp           # macOS

# Connect and upload
ncftp -u user455 -p SUpJtMdfpOvFC files.gigadb.org
ncftp> mkdir 5B
ncftp> mkdir 3C
ncftp> cd 5B
ncftp> put /home/thibaut/mahdi/models/5B/config_and_model.pth
ncftp> cd ../3C
ncftp> put /home/thibaut/mahdi/models/3C/config_and_model.pth
ncftp> quit
```

---

## After Upload: Generate MD5 Checksums

### Method 1: On Local Server (Before Upload)

```bash
# Generate checksums before upload
cd /home/thibaut/mahdi/models
md5sum 5B/config_and_model.pth > md5sums.txt
md5sum 3C/config_and_model.pth >> md5sums.txt

# View checksums
cat md5sums.txt

# Upload md5sums.txt to FTP
# Using command line FTP:
ftp files.gigadb.org
# login as user455
ftp> put /home/thibaut/mahdi/models/md5sums.txt
ftp> bye
```

### Method 2: Ask GigaDB to Generate (After Upload)

If you can't generate checksums locally, you can ask the GigaDB team to generate them on the server side after upload. Mention this in your email to `database@gigasciencejournal.com`.

---

## Create and Upload readme.txt

### Step 1: Create readme.txt

```bash
cat > readme.txt << 'EOF'
# ViralEmbed: Pretrained Model Weights
# Manuscript: GIGA-D-25-00436
# Extending Protein Language Models to a Viral Genomic Scale Using Biologically Induced Sparse Attention

## Files in This Directory:

### Model Weights:

1. 5B/config_and_model.pth
   Description: LV-5B model checkpoint
   Architecture: 33 layers, 20 attention heads, 1280 hidden dimension
   Max sequence length: 61,000 amino acids
   Format: PyTorch checkpoint (.pth)
   Size: ~2.5 GB
   MD5: [See md5sums.txt]
   Usage: Main model for processing viral genomes up to 61,000 amino acids
   Related manuscript figures: Figures 1-6

2. 3C/config_and_model.pth
   Description: LV-3C model checkpoint
   Architecture: 33 layers, 20 attention heads, 1280 hidden dimension
   Max sequence length: 61,000 amino acids
   Format: PyTorch checkpoint (.pth)
   Size: ~2.5 GB
   MD5: [See md5sums.txt]
   Usage: Alternative model variant with different attention patterns
   Related manuscript figures: Figures 3-6

3. md5sums.txt
   Description: MD5 checksums for file integrity verification

## All Other Data Files:

See GitHub repository: https://github.com/Hawaii-Bioinformatics/ViralEmbed

Available in repository:
- Training data: paper_data/genomes_dataset_train.fa (30 MB)
- Protein sequences: paper_data/viral.1.protein.faa (207 MB)
- Protein metadata: paper_data/viral.1.protein.gpff (1.9 GB)
- Test genomes: paper_data/100_genomes_lv.faa (294 KB)
- Validation set: paper_data/pola_rnr_hel.pkl (83 genomes)
- PPI scores: paper_data/pkl_pairs_gen5.tar.gz (6,789 genomes)
- Example data: large_context/scripts/data/
- Test files: test_files/

## Loading Instructions:

```python
import torch
from large_context.scripts.large_prot_encoding.model import SparseForTokenClassification

# Load checkpoint
checkpoint = torch.load('5B/config_and_model.pth',
                       map_location='cpu',
                       weights_only=False)

# Initialize model
model = SparseForTokenClassification(config=checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to('cuda')
model.eval()
```

## System Requirements:

- Python >= 3.9
- PyTorch >= 1.8
- GPU: NVIDIA with >= 16 GB VRAM (recommended: V100, A100, RTX 3090/4090)
- RAM: >= 32 GB
- Storage: >= 10 GB
- OS: Linux, macOS, Windows (with WSL)

## Installation:

See complete installation instructions and documentation:
https://github.com/Hawaii-Bioinformatics/ViralEmbed/blob/main/README.md

## License:

Software: MIT License (OSI-approved)
Data: CC0 1.0 Universal (Public Domain Dedication)

See LICENSE.md and LICENSE-DATA.md in GitHub repository.

## Citation:

[To be added upon publication]

## Contact:

Primary Author:
  Dr. Mahdi Belcaid
  Email: mahdi@hawaii.edu
  Institution: University of Hawaii

Co-author / Technical Contact:
  Thibaut Dejean
  Email: thib.dejean69@gmail.com

GigaScience Data Curator:
  Email: database@gigasciencejournal.com

## Additional Information:

For detailed documentation, see:
- Repository README: https://github.com/Hawaii-Bioinformatics/ViralEmbed/blob/main/README.md
- Technical documentation: https://github.com/Hawaii-Bioinformatics/ViralEmbed/blob/main/CLAUDE.md
- Data availability answers: https://github.com/Hawaii-Bioinformatics/ViralEmbed/blob/main/answers_data_and_code.md

Last updated: October 30, 2025
EOF
```

### Step 2: Upload readme.txt

```bash
# Using FTP command line:
ftp files.gigadb.org
# login as user455
ftp> put readme.txt
ftp> bye

# Or using lftp:
lftp -c "
open -u user455,SUpJtMdfpOvFC files.gigadb.org
put readme.txt
bye
"
```

---

## Verification

### After Upload, Verify Files:

```bash
# Connect to FTP
ftp files.gigadb.org
# login as user455

# List files
ftp> ls -lh
ftp> cd 5B
ftp> ls -lh
ftp> cd ../3C
ftp> ls -lh
ftp> cd ..

# Verify file sizes match expectations:
# - 5B/config_and_model.pth: ~2.5 GB
# - 3C/config_and_model.pth: ~2.5 GB
# - readme.txt: ~3 KB
# - md5sums.txt: ~100 bytes

ftp> bye
```

---

## Email Notification to Editors

After upload is complete, send email to:
- **To:** editorial@gigasciencejournal.com
- **CC:** database@gigasciencejournal.com

**Subject:** Data Upload Complete - GIGA-D-25-00436

**Email Template:**

```
Dear GigaScience Editors,

We have completed the data upload for manuscript GIGA-D-25-00436
"Extending Protein Language Models to a Viral Genomic Scale Using Biologically Induced Sparse Attention"

FTP Upload Details:
- Server: files.gigadb.org
- Username: user455
- Files uploaded:
  * 5B/config_and_model.pth (~2.5 GB) - LV-5B model weights
  * 3C/config_and_model.pth (~2.5 GB) - LV-3C model weights
  * readme.txt - File descriptions and usage instructions
  * md5sums.txt - MD5 checksums for integrity verification

GitHub Repository:
- URL: https://github.com/Hawaii-Bioinformatics/ViralEmbed
- Status: Public, with OSI-approved license (MIT)
- Contents: All code, training/test data, documentation
- Data license: CC0 1.0 for data files

Summary:
- Total data size: ~7.15 GB (5 GB on FTP, 2.15 GB on GitHub)
- All processed data available
- Complete documentation provided
- Reproducible workflows included

Detailed response document is available in the repository:
https://github.com/Hawaii-Bioinformatics/ViralEmbed/blob/main/answers_data_and_code.md

Next steps:
- SciCrunch RRID registration: In progress
- DOME-ML annotation: In progress

Please let us know if you need any additional information or encounter any issues accessing the data.

Best regards,
[Your Name]
[Your Institution]
[Your Email]
```

---

## Troubleshooting

### Connection Issues

**Problem:** Can't connect to FTP server

**Solutions:**
1. Verify using standard FTP protocol (not SFTP)
2. Check firewall settings (FTP uses ports 20-21)
3. Try passive mode in FileZilla (File → Settings → Connection → FTP → Passive)
4. Contact GigaDB if still failing: database@gigasciencejournal.com

### Upload Timeout

**Problem:** Upload times out or fails for large files

**Solutions:**
1. Use `lftp` instead of standard FTP (better for large files)
2. Enable binary mode: `ftp> binary`
3. Increase timeout in client settings
4. Split upload into chunks if possible
5. Use wired connection instead of WiFi for stability

### Permission Denied

**Problem:** Cannot create directories or write files

**Solutions:**
1. Double-check username and password
2. Verify logged in correctly
3. Contact GigaDB support if permissions issue persists

### File Corruption

**Problem:** Uploaded file is corrupted or incomplete

**Solutions:**
1. Ensure binary mode is enabled (`ftp> binary`)
2. Verify checksums match between local and uploaded files
3. Re-upload if checksums don't match
4. Use `lftp` which has better error handling

---

## Checklist

Before sending notification email:

- [ ] Successfully connected to FTP server
- [ ] Created directories: 5B/ and 3C/
- [ ] Uploaded 5B/config_and_model.pth (~2.5 GB)
- [ ] Uploaded 3C/config_and_model.pth (~2.5 GB)
- [ ] Generated md5sums.txt with checksums
- [ ] Uploaded md5sums.txt
- [ ] Created and uploaded readme.txt
- [ ] Verified all files present with `ls` command
- [ ] Verified file sizes are correct
- [ ] Logged out of FTP

After upload:

- [ ] Sent notification email to editors
- [ ] Kept backup of uploaded files locally
- [ ] Saved copy of MD5 checksums
- [ ] Updated GitHub repository if needed

---

## Support Contacts

**Technical Issues with FTP:**
- Email: database@gigasciencejournal.com

**General Questions:**
- Email: editorial@gigasciencejournal.com

**Author Contacts:**
- Mahdi Belcaid: mahdi@hawaii.edu
- Thibaut Dejean: thib.dejean69@gmail.com

---

**Document Created:** October 30, 2025
**Upload Deadline:** 3 days from October 22, 2025
**Priority:** URGENT
