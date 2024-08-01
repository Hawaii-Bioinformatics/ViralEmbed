import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import LayerNorm
from typing import Optional, Tuple, List, Union

from bio_attention import reshape_tensor_padding, transpose_for_scores_padding, chunk_proteins_padding, rows_to_crows, sparse_attention_matrix_padding, sparse_bsr_dropout, revert_padding
from bio_attention import generate_list, generate_couples
import numpy as np
from torch.sparse._triton_ops import bsr_softmax, bsr_dense_mm
from pos_embeddings import RotaryEmbedding
from transformers.modeling_outputs import ModelOutput, MaskedLMOutput, BaseModelOutputWithPoolingAndCrossAttentions, TokenClassifierOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers import PreTrainedModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from config_model import SparseConfig
import os
import json
import math
from memory_profiler import profile
from two_step_attention import rank_assembled_pairs
from transformers.pytorch_utils import apply_chunking_to_forward

def gpu_memory_usage(p = True):
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        used_memory = torch.cuda.memory_allocated(0)
        free_memory = total_memory - used_memory
        memory_usage = (used_memory / total_memory) * 100

        if p : 
            print(f"Utilisé: {used_memory / (1024 ** 2):.2f} MB, Libre: {free_memory / (1024 ** 2):.2f} MB, Utilisation: {memory_usage:.2f}%")
        return(memory_usage)
    else:
        print("CUDA n'est pas disponible")

class SparseLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = torch.nn.functional.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias
        return x
    
# SelfOutput class - Dense + Dropout + Add input
class SparseSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor): 
        hidden_states = hidden_states.to(self.dense.weight.device)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states
    
# Attention class - LayerNorm + SelfAttention + SelfOutput
class SparseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = SparseSelfAttention(config)
        self.output = SparseSelfOutput(config)
        self.pruned_heads = set()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    
    def forward(
        self,
        hidden_states,
        proteins_sizes = None, 
        proteins_interactions = None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        two_step_selection=False,
        device = None, 
        init_device = None
    ):
        self.device = device
        hidden_states_ln = self.LayerNorm(hidden_states)
        block_size = 32

        self_outputs = self.self_attention(
            hidden_states_ln,
            proteins_sizes=proteins_sizes,  # Assurez-vous que l'argument est nommé correctement ici
            proteins_interactions=proteins_interactions,
            block_size = block_size,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            two_step_selection=two_step_selection,
            device = device
        )
        hidden_states.to(init_device)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

# Self Attention class - Sparse attention method
class SparseSelfAttention(nn.Module) : 

    def __init__(self, config, position_embedding_type=None):
        super().__init__()

        assert config.hidden_size % config.num_attention_heads == 0

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        #self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
        self.position_embedding_type = "rotary"

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
            
        elif self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size*self.num_attention_heads)

        self.is_decoder = config.is_decoder

    def stat_attention(self, attentions, t1, t2):
        ans = []
        head = attentions[0]
        n = attentions.shape[2]

        idx = torch.arange(n)
        idx_i, idx_j = torch.meshgrid(idx, idx)
        
        abs_diff = (idx_i - idx_j).abs()
        
        mask_t1 = abs_diff < t1
        mask_t2 = (abs_diff >= t1) & (abs_diff < t2)
        mask_t3 = abs_diff >= t2

        for head in attentions[0]:
            t1_values = head[mask_t1]
            t2_values = head[mask_t2]
            t3_values = head[mask_t3]

            run = (
                t1_values.mean().item(),
                t2_values.mean().item(),
                t3_values.mean().item()
            )
            ans.append(run)
        return ans

    def forward(
        self,
        hidden_states : torch.Tensor,
        proteins_sizes : Optional[torch.Tensor],
        proteins_interactions : Optional[torch.Tensor], 
        block_size: Optional[int] = 1000,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        two_step_selection: bool = False,
        key_padding_mask: Optional[torch.LongTensor] = None,
        device : torch.device = None,
    ) :
        bsz, q_len, hidden_dim = hidden_states.size()
        # To extract the attention : use proteins_sizes, proteins_interactions -> Attention row (attention for 1 protein to all proteins)
        if proteins_interactions is not None :
            proteins_sizes_cum = self.cumulative_sum(proteins_sizes.tolist())
            start_0 = proteins_sizes_cum[int(proteins_interactions[0])]+1
            end_0 = proteins_sizes_cum[int(proteins_interactions[0])+1]+1
            start_1 = proteins_sizes_cum[int(proteins_interactions[1])]+1
            end_1 = proteins_sizes_cum[int(proteins_interactions[1])+1]+1

        # Positional encoding & Linear layers
        key_layer = self.key(hidden_states)
        query_layer = self.query(hidden_states)
        value_layer = self.value(hidden_states)
        query_layer = query_layer * self.attention_head_size**-0.5
        query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)
        
        key_layer = self.transpose_for_scores(key_layer.squeeze(0))
        query_layer = self.transpose_for_scores(query_layer.squeeze(0))
        value_layer = self.transpose_for_scores(value_layer)

        if q_len > 12000 : 
            if proteins_interactions is not None : 
                outputs = self.compute_attention_block_softmax(query_layer, key_layer, value_layer, start_0, end_0, start_1, end_1, proteins_interactions,proteins_sizes=proteins_sizes.tolist(),output_attentions=output_attentions,two_step_selection=two_step_selection)
            else :
                outputs = self.compute_attention_block_softmax(query_layer, key_layer, value_layer,output_attentions=output_attentions,two_step_selection=two_step_selection,proteins_sizes=proteins_sizes.tolist())

        else : 
            spmd = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_probs = nn.functional.softmax(spmd, dim=1)
            context_layer = torch.matmul(attention_probs.to(value_layer.dtype), value_layer)
            
            x0, x1, x2, x3 = context_layer.shape
            chunked_context_layer = context_layer.view(x0, x2, x1, x3).contiguous()
            new_chunked_context_layer_shape = chunked_context_layer.size()[:-2] + (hidden_dim,)
            chunked_context_layer = chunked_context_layer.view(new_chunked_context_layer_shape)

            if output_attentions and two_step_selection : 
                assert(proteins_sizes is not None)
                mat = attention_probs.sum(dim=1)
                _, ranked_interactions_dic = rank_assembled_pairs(proteins_sizes.tolist(),mat)
                outputs = (chunked_context_layer,ranked_interactions_dic)
                

            elif output_attentions and proteins_interactions is not None : 
                extracted_attention = attention_probs[:,:,start_0:end_0,start_1:end_1] + attention_probs[:,:,start_1:end_1,start_0:end_0].transpose(-1, -2)
                outputs = (chunked_context_layer, extracted_attention)

            else :
                outputs = (chunked_context_layer, attention_probs) if output_attentions else (chunked_context_layer,)

        return outputs
    
    def reshape_output(self, context_layer, hidden_dim):
            x0, x1, x2, x3 = context_layer.shape
            chunked_context_layer = context_layer.view(x0, x2, x1, x3).contiguous()
            new_chunked_context_layer_shape = chunked_context_layer.size()[:-2] + (hidden_dim,)
            chunked_context_layer = chunked_context_layer.view(new_chunked_context_layer_shape)
            return chunked_context_layer
    
    def compute_attention_block(self, query_layer, key_layer, value_layer, start=None, end=None, proteins_interactions=None, block_size=1000, output_attentions=False,two_step_selection=False):
        total_size = query_layer.size(2)
        attention_probs_list = []
        context_layer_list = []

        key = key_layer.transpose(-1, -2)
        batch_size, num_heads, m, k = query_layer.shape
        n = key.shape[3]

        if proteins_interactions is not None : 
            start_q, start_r = start//block_size , start%block_size
            end_q, end_r = end//block_size, end%block_size
            column = []
            row = []
            started, finished = False, False

        idx = 0
        for j in range(0, n, block_size):
            j_end = min(j + block_size, n)
            block1 = query_layer[:, :, j:j_end, :]

            attention = torch.matmul(block1, key)
            attention_probs = nn.functional.softmax(attention, dim=-1)
            context_layer = torch.matmul(attention_probs.to(value_layer.dtype), value_layer)
            context_layer_list.append(context_layer)

            if proteins_interactions is not None : 
                if idx == start_q and idx == end_q : 
                    row = attention_probs[:,:,start_r:end_r,:]
                elif idx == start_q : 
                    row.append(attention_probs[:,:,start_r:,:])
                    started = True
                elif started == True and finished == False : 
                    row.append(attention_probs[:,:,:,:])
                elif idx == end_q : 
                    row = row.append(attention_probs[:,:,:end_r,:])
                    finished = True
                column.append(attention_probs[:,:,:,start:end])

            if output_attentions : 
                attention_probs_list.append(attention_probs)

            idx +=1
        
        context_layer = torch.cat(context_layer_list, dim=2)

        context_layer_reshaped = self.reshape_output(context_layer, hidden_dim=k*num_heads)
        if output_attentions : 
            out_attention = torch.cat(attention_probs_list, dim=1)

        if proteins_interactions is not None : 
            extracted_attention = torch.cat(row, dim=2) + torch.cat(column, dim=2).transpose(-1, -2)
            outputs = (context_layer_reshaped, extracted_attention)
        else :
            outputs = (context_layer_reshaped, out_attention) if output_attentions else (context_layer_reshaped,)

        return outputs
    
    def calculate_intermediate_softmax(self, block):
        max_block = torch.max(block, dim=1, keepdim=True).values
        e_x_block = torch.exp(block - max_block)
        return e_x_block, max_block

    def compute_attention_block_softmax(self, query_layer, key_layer, value_layer, start_0=None, end_0=None, start_1=None, end_1=None, proteins_interactions=None,proteins_sizes=None, block_size=1000, output_attentions=False,two_step_selection=False):
        total_size = query_layer.size(2)
        attention_probs_list = []
        context_layer_list = []
        max_list = []
        exp_list = []
        
        if output_attentions and proteins_interactions is not None : 
            start_1q, start_1r = start_1//block_size , start_1%block_size
            end_1q, end_1r = end_1//block_size, end_1%block_size
            start_0q, start_0r = start_0//block_size , start_0%block_size
            end_0q, end_0r = end_0//block_size, end_0%block_size

            column = []
            row = []
            started_r, finished_r = False, False    # rows
            started_c, finished_c = False, False    # columns

        key = key_layer.transpose(-1, -2)
        batch_size, num_heads, m, k = query_layer.shape
        n = key.shape[3]
        sum_e_x_total = torch.zeros((1, 1, block_size, m), device=query_layer.device)

        # FIRST STEP FOR GLOBAL SOFTMAX
        for j in range(0, n, block_size):
            j_end = min(j + block_size, n)
            block1 = query_layer[:, :, j:j_end, :]
            attention = torch.matmul(block1, key)
            e_x_block, _ = self.calculate_intermediate_softmax(attention)
            if e_x_block.size(2) != block_size : 
                padding = (0, 0, 0, block_size-e_x_block.size(2))
                e_x_block = F.pad(e_x_block, padding)
            sum_e_x_total += torch.sum(e_x_block, dim=1, keepdim=True)

        idx = 0
        for j in range(0, n, block_size):
            j_end = min(j + block_size, n)
            block1 = query_layer[:, :, j:j_end, :]
            attention = torch.matmul(block1, key)
            e_x_block, _ = self.calculate_intermediate_softmax(attention)
            if e_x_block.size(2) != block_size : 
                    sum_e_x_total = sum_e_x_total[:,:,:e_x_block.size(2),:]
            attention_probs = e_x_block/sum_e_x_total
            context_layer = torch.matmul(attention_probs.to(value_layer.dtype), value_layer)
            context_layer_list.append(context_layer)

            if proteins_interactions is not None : 
                # ROW
                if idx == start_0q and idx == end_0q : 
                    row.append(attention_probs[:,:,start_0r:end_0r,start_1:end_1])
                    started_r = True
                    finished_r = True
                elif idx == start_0q : 
                    row.append(attention_probs[:,:,start_0r:,start_1:end_1])
                    started_r = True
                elif idx == end_0q : 
                    row.append(attention_probs[:,:,:end_0r,start_1:end_1])
                    finished_r = True
                elif started_r == True and finished_r == False : 
                    row.append(attention_probs[:,:,:,start_1:end_1])
                # COLUMN
                if idx == start_1q and idx == end_1q : 
                    column.append(attention_probs[:,:,start_1r:end_1r,start_0:end_0])
                    started_c = True
                    finished_c = True
                elif idx == start_1q : 
                    column.append(attention_probs[:,:,start_1r:,start_0:end_0])
                    started_c = True
                elif idx == end_1q : 
                    column.append(attention_probs[:,:,:end_1r,start_0:end_0])
                    finished_c = True
                elif started_c == True and finished_c == False : 
                    column.append(attention_probs[:,:,:,start_0:end_0])

            if output_attentions and two_step_selection: 
                attention_probs_list.append(attention_probs.sum(dim=1))

            elif output_attentions and proteins_interactions is None and not two_step_selection : 
                attention_probs_list.append(attention_probs)

            idx +=1
        
        if output_attentions and two_step_selection : 
            assert(proteins_sizes is not None)
            mat = torch.cat(attention_probs_list, dim=1)
            _, ranked_interactions_dic = rank_assembled_pairs(proteins_sizes,mat)

        context_layer = torch.cat(context_layer_list, dim=2)
        context_layer_reshaped = self.reshape_output(context_layer, hidden_dim=k*num_heads)

        if output_attentions and two_step_selection : 
            outputs = (context_layer_reshaped,ranked_interactions_dic)

        elif output_attentions and proteins_interactions is not None : 
            cat_row = torch.cat(row, dim=2) if len(row) > 1 else row[0]
            cat_column = torch.cat(column, dim=2) if len(column) > 1 else column[0]

            extracted_attention = cat_row + cat_column.transpose(-1, -2)
            outputs = (context_layer_reshaped, extracted_attention)

        elif output_attentions : 
            out_attention = torch.cat(attention_probs_list, dim=2)
            outputs = (context_layer_reshaped, out_attention)

        else :
            outputs = (context_layer_reshaped,)

        return outputs

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_padding(self, x, num_attention_heads, attention_head_size):
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

    def sort_tuple_tensor(self, tensor) : 
        """
        Trie un tenseur de dimension [n, 2] en fonction de la première valeur de chaque couple.

        Args:
        tensor (torch.Tensor): Tenseur d'entrée de dimension [n, 2].

        Returns:
        torch.Tensor: Tenseur trié en fonction de la première colonne.
        """
        _, indices = torch.sort(tensor[:, 0])
        
        sorted_tensor = tensor[indices]
        
        return sorted_tensor

    def cumulative_sum(self, liste) : 
        r = 0
        res = [0]
        for l in liste : 
            r+=l
            res.append(int(r))
        return res

# Intermediate Layer class - Dense + gelu
class SparseIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states)
        return hidden_states
    
class SparseOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob, inplace=True)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_tensor
        return hidden_states

class SparseLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = 2000 # config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SparseAttention(config)
        self.is_decoder = config.is_decoder
        self.intermediate = SparseIntermediate(config)
        self.output = SparseOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        proteins_sizes = None, 
        proteins_interactions = None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        two_step_selection=False,
        device = None, 
        init_device = None
    ):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        self_attention_outputs = self.attention(
            hidden_states,
            proteins_sizes, 
            proteins_interactions, 
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            two_step_selection=two_step_selection,
            past_key_value=self_attn_past_key_value,
            device = device, 
            init_device = init_device
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        ff_chunking = True

        _, self.chunk_size_feed_forward = self.find_chunk_size(hidden_states.shape[1], 200)

        if ff_chunking : 
            layer_output = apply_chunking_to_forward(
            self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        else : 
            layer_output = self.feed_forward(attention_output)
 
        outputs = (layer_output,) + outputs

        del layer_output
        del attention_output
        #torch.cuda.empty_cache()

        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    
    def ff_chunk(self, attn_output):
        intermediate_output = self.intermediate(attn_output)
        layer_output = self.output(intermediate_output, attn_output)
        del intermediate_output
        del attn_output
        #torch.cuda.empty_cache()
        return layer_output

    def feed_forward(self, attention_output):
        attention_output_ln = self.LayerNorm(attention_output)
        intermediate_output = self.intermediate(attention_output_ln)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
    def feed_forward_chunk(self, attention_output):

        chunks = torch.split(attention_output, self.chunk_size_feed_forward, dim=self.seq_len_dim)
        layer_output = []

        for chunk in chunks:
            chunk_output = self.output(self.intermediate(self.LayerNorm(chunk)), chunk)
            layer_output.append(chunk_output)

        layer_output = torch.cat(layer_output, dim=self.seq_len_dim)
        del attention_output
        torch.cuda.empty_cache()

        return layer_output
    
    def find_chunk_size(self, n, target) : 
        l = []
        for i in range(1, n//2) : 
            if n % i  == 0 : 
                l.append(n)
        return(l, min(l, key=lambda x: abs(x - target)))

class SparseEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([SparseLayer(config) for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        proteins_sizes = None,
        proteins_interactions = None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        two_step_selection=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
        
            init_device = hidden_states.device
            
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1 : 
                index = i%num_gpus
                device = torch.device(f"cuda:{index}" if torch.cuda.is_available() else 'cpu')
            else : 
                device = torch.device(f"cuda" if torch.cuda.is_available() else 'cpu')

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None   

            layer_outputs = layer_module(
                hidden_states,
                proteins_sizes, 
                proteins_interactions,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
                two_step_selection,
                device, 
                init_device
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)
            if output_attentions:

                if two_step_selection : 
                    if all_self_attentions == () : 
                        all_self_attentions = layer_outputs[1]
                    else :
                        for pair, score in all_self_attentions.items():
                            all_self_attentions[pair]+=score

                elif proteins_interactions is not None : 
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

                else : 
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # if proteins_interactions is not None :
            #     all_self_attentions = all_self_attentions + (layer_outputs[1],)
                
        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
      
class SparseEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if config.emb_layer_norm_before:
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.layer_norm = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.padding_idx = config.pad_token_id

        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        self.token_dropout = config.token_dropout
        self.mask_token_id = config.mask_token_id

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # Note that if we want to support ESM-1 (not 1b!) in future then we need to support an
        # embedding_scale factor here.
        embeddings = inputs_embeds

        # Matt: ESM has the option to handle masking in MLM in a slightly unusual way. If the token_dropout
        # flag is False then it is handled in the same was as BERT/RoBERTa. If it is set to True, however,
        # masked tokens are treated as if they were selected for input dropout and zeroed out.
        # This "mask-dropout" is compensated for when masked tokens are not present, by scaling embeddings by
        # a factor of (fraction of unmasked tokens during training) / (fraction of unmasked tokens in sample).
        # This is analogous to the way that dropout layers scale down outputs during evaluation when not
        # actually dropping out values (or, equivalently, scale up their un-dropped outputs in training).
        if self.token_dropout:
            embeddings = embeddings.masked_fill((input_ids == self.mask_token_id).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8  # Hardcoded as the ratio used in all ESM model training runs
            src_lengths = attention_mask.sum(-1)
            mask_ratio_observed = (input_ids == self.mask_token_id).sum(-1).float() / src_lengths
            embeddings = (embeddings * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]).to(
                embeddings.dtype
            )

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        if attention_mask is not None:
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(embeddings.dtype)
        # Matt: I think this line was copied incorrectly from BERT, disabling it for now.
        # embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

class SparsePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class SparseModel(nn.Module):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__()
        self.config = config

        self.embeddings = SparseEmbeddings(config)
        self.encoder = SparseEncoder(config)

        self.pooler = SparsePooler(config) if add_pooling_layer else None


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        proteins_sizes: Optional[torch.LongTensor] = None,
        proteins_interactions: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        two_step_selection: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        
        output_attentions = output_attentions if output_attentions is not None or proteins_interactions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
        
        input_ids= input_ids

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            proteins_sizes = proteins_sizes,
            proteins_interactions = proteins_interactions,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            two_step_selection=two_step_selection,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
    
    def get_extended_attention_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = None
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        dtype = torch.float32

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

class SparseForMaskedLM(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.model = SparseModel(config, add_pooling_layer=False)
        self.lm_head = SparseLMHead(config)
        self.config = config

        # self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        proteins_sizes: Optional[torch.LongTensor] = None,
        proteins_interactions: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        two_step_selection: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """


        if return_dict is not None :
            return_dict = return_dict 
        #print('input_ids')
        #print(input_ids) 

        outputs = self.model(
            input_ids,
            proteins_sizes = proteins_sizes, 
            proteins_interactions = proteins_interactions, 
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            labels = labels.to(prediction_scores.device)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            #return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)
        
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f)

class SparseForTokenClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config
        self.model = SparseModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        proteins_sizes: Optional[torch.LongTensor] = None,
        proteins_interactions: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        two_step_selection: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            proteins_sizes=proteins_sizes,
            proteins_interactions=proteins_interactions,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            two_step_selection=two_step_selection,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        logits = sequence_output

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx
