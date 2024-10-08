import math
from operator import __add__
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from transformers.activations import ACT2FN
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel, SDPBackend


import xformers.ops as xops


# Helpful arguments mapper
backend_map = {
    SDPBackend.MATH: {
        "enable_math": True,
        "enable_flash": False,
        "enable_mem_efficient": False,
    },
    SDPBackend.FLASH_ATTENTION: {
        "enable_math": False,
        "enable_flash": True,
        "enable_mem_efficient": False,
    },
    SDPBackend.EFFICIENT_ATTENTION: {
        "enable_math": False,
        "enable_flash": False,
        "enable_mem_efficient": True,
    },
}


class PerceiverSelfAttention(nn.Module):
    """Multi-headed {cross, self}-attention. Can be used both in the encoder as well as in the decoder."""

    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
        use_flash=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        # Q and K must have the same number of channels.
        # Default to preserving Q's input's shape.
        if qk_channels is None:
            qk_channels = q_dim
        # V's num_channels determines the shape of the output of QKV-attention.
        # Default to the same number of channels used in the key-query operation.
        if v_channels is None:
            v_channels = qk_channels
        if qk_channels % num_heads != 0:
            raise ValueError(
                f"qk_channels ({qk_channels}) must be divisible by num_heads ({num_heads})."
            )
        if v_channels % num_heads != 0:
            raise ValueError(
                f"v_channels ({v_channels}) must be divisible by num_heads ({num_heads})."
            )

        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.qk_channels_per_head = self.qk_channels // num_heads
        self.v_channels_per_head = self.v_channels // num_heads

        # Layer normalization
        self.layernorm1 = nn.LayerNorm(q_dim)
        self.layernorm2 = nn.LayerNorm(kv_dim) if is_cross_attention else nn.Identity()

        # Projection matrices
        self.query = nn.Linear(q_dim, qk_channels)
        self.key = nn.Linear(kv_dim, qk_channels)
        self.value = nn.Linear(kv_dim, v_channels)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.use_flash = use_flash
        self.use_mask = config.use_mask

    def transpose_for_scores(self, x, channels_per_head):
        new_x_shape = x.size()[:-1] + (self.num_heads, channels_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        hidden_states = self.layernorm1(hidden_states)
        inputs = self.layernorm2(inputs)

        # Project queries, keys and values to a common feature dimension. If this is instantiated as a cross-attention module,
        # the keys and values come from the inputs; the attention mask needs to be such that the inputs's non-relevant tokens are not attended to.
        is_cross_attention = inputs is not None
        queries = self.query(hidden_states)

        if is_cross_attention:
            keys = self.key(inputs)
            values = self.value(inputs)
            # attention_mask = inputs_mask
            # if inputs_mask is not None:
            #     attention_mask = inputs_mask
        else:
            keys = self.key(hidden_states)
            values = self.value(hidden_states)

        # Reshape channels for multi-head attention.
        # We reshape from (batch_size, time, channels) to (batch_size, num_heads, time, channels per head)
        queries = self.transpose_for_scores(queries, self.qk_channels_per_head)
        keys = self.transpose_for_scores(keys, self.qk_channels_per_head)
        values = self.transpose_for_scores(values, self.v_channels_per_head)

        batch_size, num_heads, seq_len, q_head_dim = queries.shape
        _, _, _, v_head_dim = values.shape
        hiddens = self.num_heads * v_head_dim

        if self.use_flash:
            # xformers attention expects shape B, N, H, D instead of B, H, N, D

            queries = queries.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            
            B, Mq, H, D = queries.shape
            Mk = keys.shape[1]
            
            if self.use_mask and inputs_mask is not None:
                attn_mask = inputs_mask == 1
                n_atoms = attn_mask.sum(dim=-1)
                n_atoms = [int(x) for x in n_atoms]

                q_seqlen = [Mq for _ in range(B)]
                attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen=q_seqlen, kv_seqlen=n_atoms)

                # BlockDiagnolMask only takes batch_size=1, need reshaping
                queries = queries.view(1, -1, H, D)
                keys_, values_ = [], []
                for i, n in enumerate(n_atoms):
                    keys_.append(keys[i, :n, :, :])
                    values_.append(values[i, :n, :, :])
                keys = torch.cat(keys_, dim=0).unsqueeze(0)
                values = torch.cat(values_, dim=0).unsqueeze(0)

            elif self.use_mask and attention_mask is not None:
                attn_mask = attention_mask == 1
                n_atoms = attn_mask.sum(dim=-1)
                n_atoms = [int(x) for x in n_atoms]

                kv_seqlen = [Mk for _ in range(B)]
                attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen=n_atoms, kv_seqlen=kv_seqlen)
                
                # BlockDiagnolMask only takes batch_size=1, need reshaping
                keys = keys.view(1, -1, H, D)
                values = values.view(1, -1, H, D)
                queries_ = []
                for i, n in enumerate(n_atoms):
                    queries_.append(queries[i, :n, :, :])
                queries = torch.cat(queries_, dim=0).unsqueeze(0)

            else:
                attn_bias = None

            context_layer = xops.memory_efficient_attention(
                queries, keys, values, 
                p=float(self.dropout.p), 
                attn_bias=attn_bias,
            )

            if self.use_mask and inputs_mask is not None:
                context_layer = context_layer.view(B, Mq, H, D)
            elif self.use_mask and attention_mask is not None:
                context_layer_ = []
                curr_n = 0
                for n in n_atoms:
                    cl = context_layer[:, curr_n:curr_n+n, :, :]
                    cl = F.pad(cl, (0, 0, 0, 0, 0, Mq-n, 0, 0))
                    curr_n += n
                    context_layer_.append(cl)
                context_layer = torch.cat(context_layer_, dim=0)
            
            context_layer = context_layer.transpose(1, 2)
            attention_probs = None

        else:
            # Take the dot product between the queries and keys to get the raw attention scores.
            attention_scores = torch.matmul(queries, keys.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(q_head_dim)

            # cross-attention between input and latent arrays
            if self.use_mask and inputs_mask is not None:
                attn_mask = inputs_mask[:, None, None, :]
                attn_mask = attn_mask.to(dtype=attention_scores.dtype)  # fp16 compatibility
                attn_mask = (1.0 - attn_mask) * torch.finfo(attention_scores.dtype).min
                attention_scores = attention_scores + attn_mask.to(attention_scores.device)
            # cross-attention between output and latent arrays
            elif self.use_mask and attention_mask is not None:
                attn_mask = attention_mask[:, None, :, None]
                attn_mask = attn_mask.to(dtype=attention_scores.dtype)  # fp16 compatibility
                attn_mask = (1.0 - attn_mask) * torch.finfo(attention_scores.dtype).min
                attention_scores = attention_scores + attn_mask.to(attention_scores.device)

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, values)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hiddens,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class PerceiverAttention(nn.Module):
    """Attention module, including a dense block."""

    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
        use_query_residual=True,
        use_flash=True,
    ):
        super().__init__()
        # MultiHead attention
        if is_cross_attention and qk_channels is None:
            if config.cross_attention_shape_for_attention == "q":
                qk_channels = q_dim
            elif config.cross_attention_shape_for_attention == "kv":
                qk_channels = kv_dim
            else:
                raise ValueError(
                    f"Unknown value {config.cross_attention_shape_for_attention} for "
                    "cross_attention_shape_for_attention."
                )
        else:
            if qk_channels is None:
                qk_channels = q_dim
            if v_channels is None:
                v_channels = qk_channels
        self.self = PerceiverSelfAttention(
            config,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
            use_flash=use_flash,
        )
        # dense block
        output_channels = None
        if is_cross_attention:
            output_channels = q_dim
        else:
            if output_channels is None:
                output_channels = v_channels
        self.output = PerceiverSelfOutput(
            config, input_channels=self.self.v_channels, output_channels=output_channels
        )
        self.use_query_residual = use_query_residual
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )

        # Output projection
        attention_output = self.output(self_outputs[0])

        # Optionally include a residual to the original queries.
        # Consider omitting the residual if the semantics of query and output
        # are different, e.g. if queries are positions and outputs are pixels.
        if self.use_query_residual:
            attention_output = attention_output + hidden_states

        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class PerceiverLayer(nn.Module):
    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
        widening_factor=4,
        use_query_residual=True,
        use_flash=True,
    ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PerceiverAttention(
            config,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
            use_query_residual=use_query_residual,
            use_flash=use_flash,
        )
        self.layernorm = nn.LayerNorm(q_dim)
        self.mlp = PerceiverMLP(
            config, input_size=q_dim, widening_factor=widening_factor
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )
        attention_output = attention_outputs[0]

        outputs = attention_outputs[1:]  # add attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )

        layer_output = layer_output + attention_output  # residual connection

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        layer_output = self.layernorm(attention_output)
        layer_output = self.mlp(layer_output)
        return layer_output


class PerceiverMLP(nn.Module):
    """A Transformer-style dense module to follow attention."""

    def __init__(self, config, input_size, widening_factor):
        super().__init__()
        self.dense1 = nn.Linear(input_size, widening_factor * input_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.Linear(widening_factor * input_size, input_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states


class PerceiverSelfOutput(nn.Module):
    def __init__(self, config, input_channels, output_channels):
        super().__init__()
        self.dense = nn.Linear(input_channels, output_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return hidden_states
