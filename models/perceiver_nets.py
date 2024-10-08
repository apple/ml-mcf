import torch
from torch import nn
import numpy as np
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from transformers import PerceiverModel, PerceiverConfig
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverBasicDecoder,
    PerceiverModel,
    PerceiverModelOutput,
    PerceiverEncoder,
    PerceiverEmbeddings,
    build_position_encoding,
)

from models.perceiver_layers import PerceiverLayer


ModalitySizeType = Mapping[str, int]
PreprocessorOutputType = Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
PreprocessorType = Callable[..., PreprocessorOutputType]
PostprocessorType = Callable[..., Any]


class Perceiver(PerceiverModel):
    def __init__(
        self,
        config,
        decoder=None,
        input_preprocessor: PreprocessorType = None,
        output_postprocessor: PostprocessorType = None,
    ):
        super().__init__(
            config,
            decoder=decoder,
            input_preprocessor=input_preprocessor,
            output_postprocessor=output_postprocessor,
        )

        self.encoder = PerceiverEnc(
            config,
            kv_dim=input_preprocessor.num_channels
            if input_preprocessor is not None
            else config.d_model,
        )

        self.embeddings = PerceiverEmbeddings(config)
        self.decoder = decoder

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.latents

    def set_input_embeddings(self, value):
        self.embeddings.latents = value

    def forward(
        self,
        inputs: torch.FloatTensor,
        queries: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        subsampled_output_points: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PerceiverModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.input_preprocessor is not None:
            inputs, modality_sizes, inputs_without_pos = self.input_preprocessor(inputs)
        else:
            modality_sizes = None
            inputs_without_pos = None
            if inputs.size()[-1] != self.config.d_model:
                raise ValueError(
                    f"Last dimension of the inputs: {inputs.size()[-1]} doesn't correspond to config.d_model:"
                    f" {self.config.d_model}. Make sure to set config.d_model appropriately."
                )

        batch_size, seq_length, _ = inputs.size()
        device = inputs.device

        # If no attention mask is provided, make them all ones
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        # Make the attention mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # extended_attention_mask = self.invert_attention_mask(attention_mask)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_blocks x num_heads]
        # and head_mask is converted to shape [num_blocks x batch x num_heads x N x N]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_blocks * self.config.num_self_attends_per_block
        )

        embedding_output = self.embeddings(batch_size=batch_size)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=None,
            head_mask=head_mask,
            inputs=inputs,
            # inputs_mask=extended_attention_mask,
            inputs_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        logits = None
        if self.decoder:
            if subsampled_output_points is not None:
                output_modality_sizes = {
                    "audio": subsampled_output_points["audio"].shape[0],
                    "image": subsampled_output_points["image"].shape[0],
                    "label": 1,
                }
            else:
                output_modality_sizes = None

            if queries is None:
                queries = self.decoder.decoder_query(
                    inputs,
                    modality_sizes,
                    inputs_without_pos,
                    subsampled_points=subsampled_output_points,
                )
            decoder_outputs = self.decoder(
                queries,
                z=sequence_output,
                # query_mask=extended_attention_mask,
                query_mask=attention_mask,
                output_attentions=output_attentions,
            )
            logits = decoder_outputs.logits

            # add cross-attentions of decoder
            if output_attentions and decoder_outputs.cross_attentions is not None:
                if return_dict:
                    encoder_outputs.cross_attentions = (
                        encoder_outputs.cross_attentions
                        + decoder_outputs.cross_attentions
                    )
                else:
                    encoder_outputs = encoder_outputs + decoder_outputs.cross_attentions

            if self.output_postprocessor:
                logits = self.output_postprocessor(
                    logits, modality_sizes=output_modality_sizes
                )

        if not return_dict:
            if logits is not None:
                return (logits, sequence_output) + encoder_outputs[1:]
            else:
                return (sequence_output,) + encoder_outputs[1:]

        return PerceiverModelOutput(
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class PerceiverEnc(PerceiverEncoder):
    """The Perceiver Encoder: a scalable, fully attentional encoder."""

    def __init__(self, config, kv_dim=None):
        super().__init__(config=config, kv_dim=kv_dim)

        # Construct the cross attention layer.
        cross_attention_layers = []
        self_attention_layers = []
        for _ in range(self.config.num_blocks):
            layer = PerceiverLayer(
                config,
                is_cross_attention=True,
                qk_channels=config.qk_channels,
                v_channels=config.v_channels,
                num_heads=config.num_cross_attention_heads,
                q_dim=config.d_latents,
                kv_dim=kv_dim,
                widening_factor=config.cross_attention_widening_factor,
                use_query_residual=config.use_query_residual,
                use_flash=config.use_flash,
            )

            cross_attention_layers.append(layer)
            self.cross_attends = nn.ModuleList(cross_attention_layers)

            # Construct a single block of self-attention layers.
            # We get deeper architectures by applying this block more than once.
            _self_attention_layers_block = []
            for _ in range(config.num_self_attends_per_block):
                layer = PerceiverLayer(
                    config,
                    is_cross_attention=False,
                    qk_channels=config.qk_channels,
                    v_channels=config.v_channels,
                    num_heads=config.num_self_attention_heads,
                    q_dim=config.d_latents,
                    kv_dim=config.d_latents,
                    widening_factor=config.self_attention_widening_factor,
                    use_query_residual=True,
                    use_flash=config.use_flash,
                )
                _self_attention_layers_block.append(layer)

            self_attention_layers.append(nn.ModuleList(_self_attention_layers_block))
            self.self_attends = nn.ModuleList(self_attention_layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        if output_attentions:
            all_cross_attentions = all_cross_attentions + (layer_outputs[1],)

        # Apply the block of self-attention layers more than once:
        for j, cross_attend in enumerate(self.cross_attends):
            # Apply the cross-attention between the latents (hidden_states) and inputs:
            layer_outputs = cross_attend(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=None,
                inputs=inputs,
                inputs_mask=inputs_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            for i, (layer_module) in enumerate(self.self_attends[j]):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class PerceiverDec(PerceiverBasicDecoder):
    def __init__(
        self,
        config: PerceiverConfig,
        output_num_channels: int,
        pos_embed_num_channels: int,
        position_encoding_type: Optional[str] = "none",
        # The following 2 arguments are ignored if position_encoding_type == 'none':
        output_index_dims: Optional[int] = None,
        num_channels: Optional[int] = 128,
        subsampled_index_dims: Optional[int] = None,
        qk_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        num_heads: Optional[int] = 1,
        widening_factor: Optional[int] = 1,
        use_query_residual: Optional[bool] = False,
        concat_preprocessed_input: Optional[bool] = False,
        final_project: Optional[bool] = True,
        position_encoding_only: Optional[bool] = False,
        **position_encoding_kwargs,
    ) -> None:
        self.pos_embed_num_channels = pos_embed_num_channels
        super().__init__(
            config,
            output_num_channels,
            position_encoding_type,
            output_index_dims,
            num_channels,
            subsampled_index_dims,
            qk_channels,
            v_channels,
            num_heads,
            widening_factor,
            use_query_residual,
            concat_preprocessed_input,
            final_project,
            position_encoding_only,
            **position_encoding_kwargs,
        )

        self.output_num_channels = output_num_channels
        # If `none`, the decoder will not construct any position encodings.
        # You should construct your own when querying the decoder.
        self.output_position_encodings = None
        self.position_encoding_type = position_encoding_type
        self.position_encoding_kwargs = position_encoding_kwargs
        if position_encoding_type != "none":
            (
                self.output_position_encodings,
                self.positions_projection,
            ) = build_position_encoding(
                position_encoding_type=position_encoding_type,
                **position_encoding_kwargs,
            )

        self.output_index_dims = output_index_dims
        self.num_channels = num_channels
        if subsampled_index_dims is None:
            subsampled_index_dims = output_index_dims
        self.subsampled_index_dims = subsampled_index_dims
        self.concat_preprocessed_input = concat_preprocessed_input
        self.final_project = final_project
        self.position_encoding_only = position_encoding_only

        # for multimodal autoencoding, we don't need the decoder cross-attention and final layer
        # so then we will set position_encoding_only to True

        if not self.position_encoding_only:
            self.decoding_cross_attention = PerceiverLayer(
                config,
                is_cross_attention=True,
                qk_channels=qk_channels,
                v_channels=v_channels,
                num_heads=num_heads,
                q_dim=num_channels,
                kv_dim=config.d_latents,
                widening_factor=widening_factor,
                use_query_residual=use_query_residual,
                use_flash=config.use_flash,
            )

            self.final_layer = (
                nn.Linear(num_channels, output_num_channels)
                if final_project
                else nn.Identity()
            )

    def decoder_query(
        self,
        inputs,
        modality_sizes=None,
        inputs_without_pos=None,
        subsampled_points=None,
    ):
        if self.position_encoding_type == "none":  # Queries come from elsewhere
            pos_emb = inputs
        else:
            if subsampled_points is not None:
                # subsampled_points are the indices if the inputs would be flattened
                # however, the inputs aren't flattened, that's why we use unravel_index
                # to get the indices for the unflattened array
                # unravel_index returns a tuple (x_idx, y_idx, ...)
                # stack to get the [n, d] tensor of coordinates
                indices = list(
                    torch.from_numpy(x)
                    for x in np.unravel_index(
                        subsampled_points.cpu(), self.output_index_dims
                    )
                )
                pos = torch.stack(indices, dim=1)
                batch_size = inputs.shape[0]
                # Map these coordinates to [-1, 1]
                pos = -1 + 2 * pos / torch.tensor(self.output_index_dims)[None, :]
                pos = torch.broadcast_to(
                    pos[None], [batch_size, pos.shape[0], pos.shape[1]]
                )
                # Construct the position encoding.
                if self.position_encoding_type == "trainable":
                    pos_emb = self.output_position_encodings(batch_size)
                elif self.position_encoding_type == "fourier":
                    pos_emb = self.output_position_encodings(
                        self.output_index_dims,
                        batch_size=batch_size,
                        device=inputs.device,
                        pos=pos,
                    )

                # Optionally project them to a target dimension.
                pos_emb = self.positions_projection(pos_emb)
                pos_emb = torch.reshape(
                    pos_emb, [pos_emb.shape[0], -1, pos_emb.shape[-1]]
                )
            else:
                batch_size = inputs.shape[0]
                index_dims = inputs.shape[2:]

                # Construct the position encoding.
                if self.position_encoding_type == "trainable":
                    pos_emb = self.output_position_encodings(batch_size)
                elif self.position_encoding_type == "fourier":
                    pos_emb = self.output_position_encodings(
                        index_dims, batch_size, device=inputs.device
                    )

                # Optionally project them to a target dimension.
                pos_emb = self.positions_projection(pos_emb)

            if self.concat_preprocessed_input:
                if inputs_without_pos is None:
                    raise ValueError(
                        "Value is required for inputs_without_pos if concat_preprocessed_input is True"
                    )
                pos_emb = torch.cat([inputs_without_pos, pos_emb], div=-1)

        return pos_emb
