import torch
from torch import nn


class PosEmbed(nn.Module):
    def __init__(
        self,
        embed_type,
        input_num_channels,
        output_num_channels,
        num_freq,
    ):
        super().__init__()

        assert embed_type in ["none", "trainable", "fourier"]
        self.embed_type = embed_type
        self.input_num_channels = input_num_channels
        self.output_num_channels = output_num_channels
        if self.embed_type == "none":
            self.embedder = nn.Identity()
        elif self.embed_type == "trainable":
            self.embedder = nn.Sequential(
                nn.Linear(input_num_channels, output_num_channels),
                nn.SiLU(),
                nn.Linear(output_num_channels, output_num_channels),
                nn.SiLU(),
                nn.Linear(output_num_channels, output_num_channels),
            )

        elif self.embed_type == "fourier":
            embed_kwargs = {
                "include_input": True,
                "input_dims": input_num_channels,
                "max_freq_log2": num_freq - 1,
                "num_freqs": num_freq,
                "log_sampling": True,
                "periodic_fns": [torch.sin, torch.cos],
            }

            embedder_obj = FourierEmbed(**embed_kwargs)
            embed_fn = lambda x, eo=embedder_obj: eo.embed(x)
            self.output_num_channels = embedder_obj.out_dim
            self.embedder = embed_fn

    def forward(self, x):
        x = self.embedder(x)
        return x


class FourierEmbed:
    """Module for positional encodings."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs, device="cuda"):
        embedding = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        embedding = (embedding - embedding.min(dim=1, keepdim=True)[0]) / (
            embedding.max(dim=1, keepdim=True)[0]
            - embedding.min(dim=1, keepdim=True)[0]
        )
        embedding = (embedding * 2.0) - 1.0
        return embedding


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        "include_input": True,
        "input_dims": 2,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = FourierEmbed(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim
