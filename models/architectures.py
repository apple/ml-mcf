import torch
from torch import nn
from torch.nn import functional as F
from transformers import PerceiverConfig
from models.perceiver_nets import PerceiverDec, Perceiver
import math
from einops import repeat
from utils.utils import instantiate_from_config


# Helper function to compute timestep embeddings
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


########################################################################
# PerceiverIO architectures (adapted from the transformers library)
########################################################################

class PerceiverIO(nn.Module):
    def __init__(
        self,
        pos_embed_config,
        num_latents=128,
        d_latents=256,
        d_model=768,
        time_sinusoidal_dim=64,
        num_blocks=1,
        num_self_attends_per_block=4,
        num_self_attention_heads=4,
        num_cross_attention_heads=4,
        signal_num_channels=3,
        proj_dim=128,
        coord_num_channels=2,
        use_flash=True,
        pos_embed_apply="both",  # This is the dimensionality to which all the inputs are initially projected
        num_classes=None,  # If num_classes is not None, we add a class embedding to the context and query tensors
        use_mask=False,
    ):
        super().__init__()

        self.pos_embed_apply = pos_embed_apply
        self.pos_embed = instantiate_from_config(pos_embed_config)
        # After we know the output dimension of positional embeddings we use that as to define the proj_dim of the architectures
        proj_dim = self.pos_embed.output_num_channels

        self.time_sinusoidal_dim = proj_dim
        self.num_latents = num_latents

        perceiver_config = PerceiverConfig(
            num_latents=num_latents,
            d_latents=d_latents,
            d_model=d_model,
            num_blocks=num_blocks,
            num_self_attends_per_block=num_self_attends_per_block,
            num_self_attention_heads=num_self_attention_heads,
            num_cross_attention_heads=num_cross_attention_heads,
            max_position_embeddings=2048,
            output_attentions=False,
            hidden_act="gelu",
            use_mask=use_mask,
        )
        perceiver_config.use_flash = use_flash

        self.perceiver = Perceiver(
            perceiver_config,
            decoder=PerceiverDec(
                perceiver_config,
                pos_embed_num_channels=d_model,
                output_num_channels=d_model,
                position_encoding_type="none",  # we build our own position encoding
                num_channels=d_model,  # number of channels of position embedding queries
                num_heads=num_cross_attention_heads,
                use_query_residual=True,
                final_project=True,
                concat_preprocessed_input=True,  # Why does this need to be True?
            ),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        if num_classes is not None:
            self.num_classes = num_classes
            self.class_embed = nn.Sequential(
                nn.Embedding(num_classes, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
            )
            context_dim = proj_dim * 4
            query_dim = proj_dim * 4
        else:
            context_dim = proj_dim * 3
            query_dim = proj_dim * 3

        self.context_x_proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.context_y_proj = nn.Sequential(
            nn.Linear(signal_num_channels, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        self.query_x_proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.query_y_proj = nn.Sequential(
            nn.Linear(signal_num_channels, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        self.context_cat_proj = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.query_cat_proj = nn.Sequential(
            nn.Linear(query_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.output_fc = nn.Linear(d_model + proj_dim, signal_num_channels)

    def forward(
        self,
        context_x=None,
        context_y=None,
        t=None,
        query_x=None,
        query_y=None,
        label=None,
        attention_mask=None,
    ):
        # Compute timestep fourier embedding and projection
        t_emb = self.time_embed(timestep_embedding(t, self.time_sinusoidal_dim))

        # Project context x's (coordinates) and y's (signals)
        if self.pos_embed_apply in ["query", "both"]:
            context_x = self.pos_embed(context_x)
        context_x = self.context_x_proj(context_x)
        context_y = self.context_y_proj(context_y)

        t_emb_context = repeat(t_emb, "b c -> b n c", n=context_x.shape[1])
        context_cat = torch.cat([context_x, context_y, t_emb_context], dim=2)
        if label is not None:
            cls_emb_1d = self.class_embed(label)
            cls_emb_context = repeat(cls_emb_1d, "b c -> b n c", n=context_x.shape[1])
            context_cat = torch.cat([context_cat, cls_emb_context], dim=2)
        context_cat = self.context_cat_proj(context_cat)

        # Project query x's (coordinates) and y's (signals)
        if self.pos_embed_apply in ["query", "both"]:
            query_x = self.pos_embed(query_x)
        query_x = self.query_x_proj(query_x)
        query_y = self.query_y_proj(query_y)

        t_emb_query = repeat(t_emb, "b c -> b n c", n=query_x.shape[1])
        query_cat = torch.cat([query_x, query_y, t_emb_query], dim=2)
        if label is not None:
            cls_emb_query = repeat(cls_emb_1d, "b c -> b n c", n=query_x.shape[1])
            query_cat = torch.cat([query_cat, cls_emb_query], dim=2)
        query_cat = self.query_cat_proj(query_cat)

        # Call Perceiver with context and query tensors
        query_pred = self.perceiver(
            inputs=context_cat,
            queries=query_cat,
            attention_mask=attention_mask,
            output_attentions=False,
        )
        query_pred = query_pred.logits
        query_pred = self.output_fc(torch.cat([query_pred, query_y], dim=2))

        return query_pred