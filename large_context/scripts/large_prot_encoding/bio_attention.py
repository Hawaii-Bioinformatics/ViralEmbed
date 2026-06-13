# All functions and methods for smart blockwise sparse attentions
# Author : Thibaut Dejean

import torch
import numpy as np
import random
import torch.nn.functional as F
from torch.sparse._triton_ops import bsr_softmax, bsr_dense_mm      # Methods for sparse bsr tensors operations
from memory_profiler import profile # For debugging
import gc


# For random sequences and interactions

def generate_list(n, Amin, Amax):
    """
    Generate a list of random protein sizes 

    Args:
    n (int) : Size of the sequence. 
    Amin (int) : Minimum size of a protein. 
    Amax (int) : Maximum size of a protein. 

    Returns:
    ni_list (list) : List of random protein sizes
    """

    ni_list = []
    remaining = n

    while remaining > Amin:
        ni = random.randint(Amin, min(Amax, remaining))
        ni_list.append(ni)
        remaining -= ni

    ni_list.append(remaining)

    return ni_list

def generate_couples(n_couples, n_len):
    """
    Generate a list of random protein/protein interactions. 

    Args:
    n_couples (int) : Number of interactions for the entire genom (pass n_couples_single*len(proteins_sizes) to chose the number of interactions for each protein)
    n_len (int) : Number of proteins in the sequence. 

    Returns:
    couples (list) : List of random protein/protein interactions. 
    """
    
    couples = []
    while len(couples) < n_couples :
        i = random.randint(0, n_len - 1)
        j = random.randint(0, n_len - 1)

        if abs(i - j) > 1 and (i, j) not in couples and (j, i) not in couples:
            couples.append((i, j))

    return couples

# For sparse attention 

def reshape_tensor_padding(tensor, proteins_sizes, block_size):
    """
    Reshape a tensor into blocks of constant size, taking into account the size of the proteins.
    If protein_size % block_size > 0 : Padding for a block of size block_size.

    Args:
        tensor (torch.Tensor) : Input tensor to reshape.
        proteins_sizes (list) : List of all the sizes of the proteins of the sequence. 
        block_size (int) : Size of a block, must be a power of 2, and greater than 16 (usually 64). 

    Returns:
        result_tensor (torch.Tensor) : The reshaped tensor
        padding_storage (list) : For each protein of the genom (index, number of blocks, number of padded values)
    """
    _, _, hidden_dim = tensor.shape
    total_blocks = sum((size + block_size - 1) // block_size for size in proteins_sizes)  # Calculate total blocks needed
    result_tensor = torch.zeros((1, total_blocks, block_size, hidden_dim), dtype=tensor.dtype, device=tensor.device)

    padding_storage = []
    
    block_index = 0
    start_index = 0

    for i, size in enumerate(proteins_sizes):
        num_full_blocks = size // block_size  
        end_index = start_index + num_full_blocks * block_size

        if num_full_blocks > 0:
            result_tensor[0, block_index:block_index + num_full_blocks] = tensor[0, start_index:end_index].reshape(num_full_blocks, block_size, hidden_dim)
            block_index += num_full_blocks
        
        remainder = size % block_size
        padding_storage.append((i, num_full_blocks, remainder))
        
        if remainder > 0:
            remainder_block = tensor[0, end_index:end_index + remainder]
            if remainder < block_size:
                padding = torch.zeros((block_size - remainder, hidden_dim), dtype=tensor.dtype, device=tensor.device)
                remainder_block = torch.cat([remainder_block, padding], dim=0)
            result_tensor[0, block_index] = remainder_block
            block_index += 1
        
        start_index += size
            
    return result_tensor, padding_storage

def transpose_for_scores_padding(x, num_attention_heads, attention_head_size):
    """
    Reshape a tensor of shape (batchsize, num_blocks, block_size ,hidden_dim) 
    to shape (batchsize, num_attention_heads, block_size, num_blocks, attention_head_size).
    Applied to Query, Key and Value before attention calculation. 

    Args:
    x (torch.Tensor) : Input tensor to reshape.
    num_attention_heads (int) : Number of attention heads (20 for ESM2-650M).
    attention_head_size (int) : Size of an attention head (64 for ESM2-650M). 

    Returns:
    x_permuted (torch.Tensor) : The reshaped tensor of shape (batchsize, num_attention_heads, block_size, num_blocks, attention_head_size)
    """
        
    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
    x = x.view(new_x_shape)
    x_permuted = x.permute(0, 3, 2, 1, 4)
    return x_permuted

def chunk_proteins_padding(sorted_proteins_interactions, proteins_lengths, block_size, device):
    """
    Transforms protein/protein interactions to block/block interaction 

    Args:
    sorted_proteins_interactions (list) : List of all interactions between proteins. 
    proteins_lengths (list) : List of all protein sizes. 
    block_size (int) : Size of a block, must be a power of 2, and greater than 16 (usually 64). 

    Returns:
    block_interactions (list) : List of all interactions between blocks. 
    chunked_blocks (list) : List of indices for all blocks. 
    """

    sorted_proteins_interactions = sorted_proteins_interactions.detach().cpu()
    proteins_lengths = proteins_lengths.detach().cpu()
    num_blocks = [(length + block_size - 1) // block_size for length in proteins_lengths]  
    num_blocks_cs = np.cumsum([0] + num_blocks).tolist()
    index = 0
    chunked_blocks = []
    
    for h in num_blocks:
        for j in range(h):
            chunked_blocks.append(index)
            index += 1

    chunked_blocks_tensor = torch.tensor(chunked_blocks, device=device)
    block_interactions = []
    
    for interaction in sorted_proteins_interactions:
        i, j = interaction[0], interaction[1]
        for k in range(num_blocks[i]):
            for h in range(num_blocks[j]):
                block_interactions.append((num_blocks_cs[i] + k, num_blocks_cs[j] + h))

    block_interactions_tensor = torch.tensor(block_interactions, device=device)

    return block_interactions_tensor, chunked_blocks_tensor

def rows_to_crows(rows, n): 
    """
    Creates a list for storing block positional information in torch.sparse_bsr_tensor format

    Args:
    rows (list) : Row of each block.
    n (int) : Number of blocks. 

    Returns:
    counts_cs (list) : list of crows_indices for torch.sparse_bsr_tensor creation
    """
    counts = torch.bincount(rows, minlength=n+1)
    
    if counts.size(0) < n+1:
        counts = torch.nn.functional.pad(counts, (0, n+1 - counts.size(0)), mode='constant', value=0)
    
    counts_cs = counts.cumsum(dim=0)
    zero_tensor = torch.zeros(1, dtype=counts_cs.dtype, device = counts_cs.device)
    count_tensor = torch.cat((zero_tensor, counts_cs), dim=0)
    
    return count_tensor

def sparse_attention_matrix_padding(query, key, chunked_interactions_padding, blocks_cs, block_size, device, tril = False):
    """
    Create the attention sparse matrix (torch.sparse_bsr_tensor format)

    Args:
    query (torch.Tensor) 
    Key (torch.Tensor) 
    sorted_blocks_interactions (list) : List of all the interactions between blocks (transformed from protein interactions by chunk_proteins_padding function). 
    blocks_cs (list) : Cumulative Sum of blocks sizes. 
    block_size (int) : Size of a block (used for assertion, can be removed)
    device (torch.device) : gpu !
    tril (bool) : If set to True, only the attention blocks in the lower triangle are calculated (to be optimized). 

    Returns:
    sparse_matrix (torch.Tensor) : The bsr sparse attention tensor, with : 
        - size = (num_heads, num_blocks*block_size, num_blocks*block_size)
        - device = device
        - nnz = len(blocks_interactions) if not tril
        - layout = torch.sparse_bsr
    """
    batch_size, num_heads, bloc_size_exp, num_blocks, all_head_size = query.shape

    assert bloc_size_exp == block_size 

    block_positions = []

    if tril : 
        chunked_interactions_padding = [interaction for interaction in chunked_interactions_padding if interaction[0] >= interaction[1]]

    #print(sorted_blocks_interactions)

    # for i, j in sorted_blocks_interactions :
        
    #     query_block = query[:, :, :,i, :]
    #     key_block = key[:, :, :,j, :]
        
    #     attention_block = torch.matmul(query_block,key_block.transpose(-1, -2)).squeeze()
    #     attentions.append(attention_block) 
    #     del query_block, key_block, attention_block
        
    query = query.reshape(batch_size, num_heads, num_blocks, bloc_size_exp, all_head_size)
    key = key.reshape(batch_size, num_heads, num_blocks, bloc_size_exp, all_head_size)
    key = key.transpose(-1,-2)

    i_indices = chunked_interactions_padding[:, 0]
    j_indices = chunked_interactions_padding[:, 1]

    attentions3 = torch.matmul(query[:, :, i_indices, :, :],key[:, :, j_indices, :, :]).squeeze()
    gc.collect()

    # Columns and Rows/Crows :     
    rows = chunked_interactions_padding[:, 0]
    col_indices = chunked_interactions_padding[:, 1]
    crow_indices = rows_to_crows(rows, len(blocks_cs) - 1)[:-1] 

    crow_tensor = crow_indices.repeat(20, 1)
    col_tensor = col_indices.repeat(20, 1)

    # sparse_matrix = create_sparse_coo_with_variable_blocks(attentions, block_positions, seq_len, seq_len) 
    sparse_matrix = torch.sparse_bsr_tensor(crow_tensor, col_tensor, attentions3, size = [num_heads, num_blocks*block_size, num_blocks*block_size])
    del attentions3, col_indices, crow_indices, rows
    gc.collect()

    return sparse_matrix

def sparse_bsr_dropout(x, p, training):
    """
    Apply dropout to a bsr sparse tensor (not inplace, highly memory consumming) 

    Args:
    x (torch.Tensor) : Input tensor.
    p (float) : Dropout parameter. 
    training (bool) : Training bool parameter for dropout 

    Returns:
    new_sparse_tensor (torch.Tensor) : Output tensor, with dropour applied
    """

    values = x.values()  
    dropped_values = F.dropout(values, p=p, training=training)  
    new_sparse_tensor = torch.sparse_bsr_tensor(x.crow_indices(), x.col_indices(), dropped_values, size=x.size())
    return new_sparse_tensor

def revert_padding(chunked_context_layer, padding_storage, block_size):
    """
    Reshape a tensor into after attention process, to get teh correct shape for the output tensor. 
    Output tensor should be of shape (batchsize, seq_length, hidden_dim)

    Args:
    chunked_context_layer (torch.Tensor) : Input tensor to reshape.
    padding_storage (list) : List of all the paddings applied (from reshape_tensor_padding function)
    block_size (int) : Size of a block, must be a power of 2, and greater than 16 (usually 64). 

    Returns:
    result (torch.Tensor) : The reshaped tensor of shape (batchsize, seq_length, hidden_dim), with the attention output. 
    """

    values = []
    start_index = 0

    for (i, num_blocks, padding) in padding_storage :
        end_index = start_index + (num_blocks * block_size) + padding 
        if padding == 0 :
            complete = (block_size * num_blocks) - 1
        else : 
            complete = block_size * (num_blocks+1) - 1

        keep = chunked_context_layer[0][start_index:end_index]
        start_index += complete
        values.append(keep)

    result = torch.cat(values, dim=0).unsqueeze(0)

    return result


### CPU RAM REDUCING ATTEMPTS


def reshape_tensor_padding_ram(tensor, proteins_sizes, block_size):
    """
    Reshape a tensor into blocks of constant size, taking into account the size of the proteins.
    If protein_size % block_size > 0 : Padding for a block of size block_size.

    Args:
        tensor (torch.Tensor) : Input tensor to reshape.
        proteins_sizes (list) : List of all the sizes of the proteins of the sequence. 
        block_size (int) : Size of a block, must be a power of 2, and greater than 16 (usually 64). 

    Returns:
        result_tensor (torch.Tensor) : The reshaped tensor
        padding_storage (list) : For each protein of the genom (index, number of blocks, number of padded values)
    """
    bsz, _, hidden_dim = tensor.shape
    total_blocks = sum((size + block_size - 1) // block_size for size in proteins_sizes)  # Calculate total blocks needed

    padding_storage = []
    padded_blocks = []
    
    block_index = 0
    start_index = 0

    for i, size in enumerate(proteins_sizes):
        num_full_blocks = size // block_size  
        protein = tensor[:, start_index:start_index + size]

        # Calculate number of blocks needed for this protein
        num_blocks = (size + block_size - 1) // block_size

        # Extend the protein to fit exactly into the blocks
        if protein.shape[1] % block_size != 0:
            padding_size = (num_blocks * block_size) - protein.shape[1]
            padding = torch.zeros(bsz, padding_size, hidden_dim, dtype=tensor.dtype, device=tensor.device)
            protein = torch.cat([protein, padding], dim=1)

        # Reshape into blocks
        protein_blocks = protein.view(bsz, num_blocks, block_size, hidden_dim)
        padded_blocks.append(protein_blocks)

        # Update the start index for the next protein
        start_index += size
        remainder = size % block_size
        padding_storage.append((i, num_full_blocks, remainder))

        max_blocks = max([b.shape[1] for b in padded_blocks])
        for k in range(len(proteins_sizes)):

            current_blocks = padded_blocks[k].shape[1]
            if current_blocks < max_blocks:
                padding = torch.zeros(bsz, max_blocks - current_blocks, block_size, hidden_dim, dtype=tensor.dtype, device=tensor.device)
                padded_blocks[k] = torch.cat([padded_blocks[k], padding], dim=1)

        # Stack all block tensors
        output_tensor = torch.cat(padded_blocks, dim=1)
        
        return output_tensor

# Example usage
# tensor = torch.randn((1, 1000, 128), device='cuda:0')  # Example input tensor
# proteins_sizes = [300, 400, 300]  # Example protein sizes
# block_size = 64

#device = 'cuda:0'
#tensor.to(device)

# result_tensor, padding_storage = reshape_tensor_padding_ram(tensor, proteins_sizes, block_size)
# print(result_tensor.shape)
# print(padding_storage)


#result_tensor_ram, padding_storage_ram = reshape_tensor_padding_ram(tensor, proteins_sizes, block_size)

def protein_block_padder(tensor, proteins_sizes, block_size):
    bsz, seq_len, hidden_dim = tensor.shape
    num_proteins = len(proteins_sizes)
    # List to hold the output blocks for all proteins
    padded_blocks = []
    padding_storage = []
    
    # Start index for slicing
    start_idx = 0

    for p, size in enumerate(proteins_sizes):
        num_full_blocks = size // block_size  
        # Slice the tensor to get the current protein
        #protein = tensor[:, start_idx:start_idx + size]

        # Calculate number of blocks needed for this protein
        num_blocks = (size + block_size - 1) // block_size

        # Extend the protein to fit exactly into the blocks
        if size % block_size != 0:
            padding_size = (num_blocks * block_size) - size
            #padding = torch.zeros(bsz, padding_size, hidden_dim, dtype=tensor.dtype, device=tensor.device)
            #protein = torch.cat([tensor[:, start_idx:start_idx + size], padding], dim=1)

            padding_F = (0,0,padding_size,0)
            protein = F.pad(tensor[:, start_idx:start_idx + size], padding_F, mode='constant', value=0).view(bsz, num_blocks, block_size, hidden_dim)

        else : 
            protein = tensor[:, start_idx:start_idx + size].view(bsz, num_blocks, block_size, hidden_dim)
        # Reshape into blocks
        #protein_blocks = protein.view(bsz, num_blocks, block_size, hidden_dim)
        padded_blocks.append(protein)

        # Update the start index for the next protein
        start_idx += size

        remainder = size % block_size
        padding_storage.append((p, num_full_blocks, remainder))

    # Concatenate all protein blocks along the block dimension
    # Since the number of blocks may vary, we take the maximum number required
    max_blocks = max([b.shape[1] for b in padded_blocks])
    # Pad other proteins to have the same number of blocks
    for i in range(num_proteins):
        current_blocks = padded_blocks[i].shape[1]
        if current_blocks < max_blocks:
            padding = torch.zeros(bsz, max_blocks - current_blocks, block_size, hidden_dim, dtype=tensor.dtype, device=tensor.device)
            padded_blocks[i] = torch.cat([padded_blocks[i], padding], dim=1)

    # Stack all block tensors
    output_tensor = torch.cat(padded_blocks, dim=1)
    
    return output_tensor, padding_storage

# result_tensor, padding_storage = protein_block_padder(tensor, proteins_sizes, block_size)
# print(result_tensor.shape)
# print(padding_storage)