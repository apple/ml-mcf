import os
import gc
import pickle
from re import T
import torch
import numpy as np
from einops import reduce, repeat
import lightning.pytorch as pl
import torch.distributed as tdist

from utils.utils import instantiate_from_config
from utils.lr_scheduler import CosineAnnealingWarmupRestarts

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm
from utils.utils import ModelEmaV2

from utils.utils import make_beta_schedule, extract, noise_like, default
from models._dpfmetrics import _DPFMetrics
from utils.metrics import set_rdmol_positions
import pyvista as pv

pv.start_xvfb()


# We pull out metrics to a separate file
class MCF(pl.LightningModule, _DPFMetrics):
    def __init__(
        self,
        architecture_config,
        sampling_config,
        loss_config,
        train_config,
        input_coord_num_channels=2,
        input_signal_num_channels=3,
        npoints_context=256,
        npoints_query=256,
        viz_dir=None,
        online_sample=True,
        online_evaluation=True,
        randperm=False,
        threshold=0.5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = instantiate_from_config(architecture_config)
        self.model_ema = ModelEmaV2(self.model)
        self.loss = instantiate_from_config(loss_config)
        self.train_config = train_config
        self.sampling_fn = sampling_config.sampling_fn
        self.num_timesteps_ddim = sampling_config.num_timesteps_ddim
        self.eta_ddim = sampling_config.eta_ddim
        self.online_sample = online_sample
        self.online_evaluation = (online_evaluation and online_sample)
        self.randperm = randperm
        self.threshold = threshold

        self.viz_dir = viz_dir
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)

        self.npoints_context = npoints_context
        self.npoints_query = npoints_query
        self.input_coord_num_channels = input_coord_num_channels
        self.input_signal_num_channels = input_signal_num_channels

        self.fid = FrechetInceptionDistance(
            feature=2048, reset_real_features=False
        ).cuda()
        self.kid = KernelInceptionDistance(
            feature=2048, reset_real_features=False, subset_size=4
        ).cuda()
        self.inception_score = InceptionScore().cuda()
        self.pv_plotter = pv.Plotter(off_screen=True)
        self.gt_pcs = []
        self.sampled_pcs = []
        self.manifold_gt_signal = []
        self.manifold_sampled_signal = []

        # Define betas (variance schedule)
        betas = make_beta_schedule(
            self.train_config.beta_schedule, self.train_config.num_timesteps
        )
        # Get total number of time steps.
        timesteps = betas.shape[0]
        self.num_timesteps = int(timesteps)

        # Define alphas: alphas are  (1 - betas), and alphas_cumprod are used to directly compute the noise at time step t.
        # Based on the "nice" property.
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64), alphas_cumprod[:-1]), 0
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.register("betas", betas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register("log_one_minus_alphas_cumprod", torch.log(1 - alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", torch.rsqrt(alphas_cumprod))
        self.register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1 / alphas_cumprod - 1))
        self.register("posterior_variance", posterior_variance)
        self.register(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register(
            "posterior_mean_coef1",
            (betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)),
        )
        self.register(
            "posterior_mean_coef2",
            ((1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)),
        )

    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def q_sample(self, y_0, t, noise=None):
        """Forward diffusion process.

        :param x_0: input at time 0 (original input).
        :param t: time step.
        :param noise: noise with the same shape as x_0, which will be added to x_0.
                    if None, gaussian noise will be used.
        :return:
            x_t, noisy image after t steps of adding noise.
        """

        if noise is None:
            noise = torch.randn_like(y_0)

        return (
            extract(self.sqrt_alphas_cumprod, t, y_0.shape) * y_0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, y_0.shape) * noise
        )

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, y_t.shape) * y_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0, y_t, t):
        mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0
            + extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        var = extract(self.posterior_variance, t, y_t.shape)
        log_var_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)

        return mean, var, log_var_clipped

    def p_mean_variance(
        self,
        context_x=None,
        context_y=None,
        t=None,
        query_x=None,
        query_y_sampled=None,
        clip_denoised=True,
        label=None,
        attention_mask=None,
    ):
        noise = self.model_ema.module(
            context_x=context_x,
            context_y=context_y,
            t=t,
            query_x=query_x,
            query_y=query_y_sampled,
            label=label,
            attention_mask=attention_mask,
        )

        query_y_0 = self.predict_start_from_noise(
            query_y_sampled,
            t,
            noise,
        )

        if clip_denoised:
            query_y_0 = query_y_0.clamp(min=-1, max=1)

        mean, var, log_var = self.q_posterior(query_y_0, query_y_sampled, t)

        return mean, var, log_var

    def p_sample(
        self,
        context_x=None,
        context_y=None,
        t=None,
        query_x=None,
        query_y=None,
        noise_fn=torch.randn,
        clip_denoised=True,
        repeat_noise=False,
        label=None,
        attention_mask=None,
    ):
        mean, _, log_var = self.p_mean_variance(
            context_x,
            context_y,
            t,
            query_x=query_x,
            query_y_sampled=query_y,
            clip_denoised=clip_denoised,
            label=label,
            attention_mask=attention_mask,
        )
        noise = noise_like(query_y.shape, noise_fn, query_y.device, repeat_noise)
        shape = [query_y.shape[0]] + [1] * (query_y.ndim - 1)
        nonzero_mask = (1 - (t == 0).type(torch.float32)).view(*shape)

        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def p_sample_loop(
        self, shape, query_x, query_y_sampled, label=None, attention_mask=None
    ):
        batch, device = shape[0], self.betas.device

        # Select subset of query pairs as context
        # idxs_sampled = torch.randperm(query_x.shape[1])[: self.npoints_context[0]]
        npoints_context = min((self.npoints_context[0], query_x.shape[1]))
        if self.randperm and attention_mask is None:
            idxs_sampled = torch.randperm(query_x.shape[1])[:npoints_context]
        else:
            idxs_sampled = torch.arange(npoints_context)
        context_x = query_x[:, idxs_sampled]

        for t in tqdm(
            reversed(range(0, self.num_timesteps)), desc="Sampling loop time step"
        ):
            context_y = query_y_sampled[:, idxs_sampled]

            query_y_sampled = self.p_sample(
                context_x=context_x,
                context_y=context_y,
                t=torch.full((shape[0],), t, dtype=torch.int64).to(device),
                query_x=query_x,
                query_y=query_y_sampled,
                noise_fn=torch.randn,
                label=label,
                attention_mask=attention_mask,
            )

            # CoM
            query_y_sampled[~attention_mask.bool()] = 0
            denom = torch.sum(attention_mask, -1, keepdim=True)
            denom = denom.unsqueeze(-1)
            y_mean = torch.sum(query_y_sampled, dim=1, keepdim=True) / denom
            query_y_sampled -= y_mean

        return query_y_sampled

    @torch.no_grad()
    def ddim_step(
        self,
        context_x=None,
        context_y=None,
        t=None,
        query_x=None,
        query_y_sampled=None,
        alpha=None,
        alpha_next=None,
        time_next=None,
        eta=None,
        clip_denoised=True,
        label=None,
        attention_mask=None,
    ):
        pred_noise = self.model_ema.module(
            context_x=context_x,
            context_y=context_y,
            t=t,
            query_x=query_x,
            query_y=query_y_sampled,
            label=label,
            attention_mask=attention_mask,
        )

        query_y_0 = self.predict_start_from_noise(query_y_sampled, t, noise=pred_noise)

        if clip_denoised:
            query_y_0.clamp_(-1.0, 1.0)

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = ((1 - alpha_next) - sigma**2).sqrt()

        noise = torch.randn_like(query_y_0) if time_next > 0 else 0.0
        query_y_sampled = query_y_0 * alpha_next.sqrt() + c * pred_noise + sigma * noise
        return query_y_sampled

    @torch.no_grad()  # On the works
    def ddim_sample(
        self,
        shape=None,
        query_x=None,
        query_y_sampled=None,
        clip_denoised=True,
        label=None,
        attention_mask=None,
    ):
        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.num_timesteps_ddim,
            self.eta_ddim,
        )

        times = torch.linspace(0.0, total_timesteps, steps=sampling_timesteps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # Set number of context points for sampline
        # sampled_indices = torch.randperm(query_x.shape[1])[: self.npoints_context[0]]
        npoints_context = min((self.npoints_context[0], query_x.shape[1]))
        if self.randperm and attention_mask is None:
            sampled_indices = torch.randperm(query_x.shape[1])[:npoints_context]
        else:
            sampled_indices = torch.arange(npoints_context)
        context_x = query_x[:, sampled_indices]

        for i, (time, time_next) in enumerate(
            tqdm(time_pairs, desc="Sampling loop time step")
        ):
            # Select subset of queries as context
            context_y = query_y_sampled[:, sampled_indices]

            alpha = self.alphas_cumprod_prev[time]
            alpha_next = self.alphas_cumprod_prev[time_next]
            t = torch.full((batch,), time, device=device, dtype=torch.long)

            query_y_sampled = self.ddim_step(
                context_x,
                context_y,
                t,
                query_x,
                query_y_sampled,
                alpha,
                alpha_next,
                time_next,
                eta,
                clip_denoised=clip_denoised,
                label=label,
                attention_mask=attention_mask,
            )

            # CoM
            query_y_sampled[~attention_mask.bool()] = 0
            denom = torch.sum(attention_mask, -1, keepdim=True)
            denom = denom.unsqueeze(-1)
            y_mean = torch.sum(query_y_sampled, dim=1, keepdim=True) / denom
            query_y_sampled -= y_mean

        return query_y_sampled

    @torch.no_grad()
    def sampling(
        self,
        shape,
        query_x,
        query_y,
        mode=None,
        label=None,
        attention_mask=None,
    ):
        if mode is None:
            mode = self.sampling_fn
        if mode == "ddim":
            query_y_sampled = self.ddim_sample(
                shape, query_x, query_y, label=label, attention_mask=attention_mask
            )
        elif mode == "standard":
            query_y_sampled = self.p_sample_loop(
                shape, query_x, query_y, label=label, attention_mask=attention_mask
            )

        return query_y_sampled

    def p_losses(
        self,
        context_x,
        context_y,
        t,
        query_x,
        query_y,
        context_noise,
        query_noise,
        label=None,
        attention_mask=None,
    ):
        context_noise = default(context_noise, lambda: torch.randn_like(context_y))
        query_noise = default(query_noise, lambda: torch.randn_like(query_y))

        context_y_t = self.q_sample(context_y, t, noise=context_noise)
        query_y_t = self.q_sample(query_y, t, noise=query_noise)

        model_out = self.model(
            context_x=context_x,
            context_y=context_y_t,
            t=t,
            query_x=query_x,
            query_y=query_y_t,
            label=label,
            attention_mask=attention_mask,
        )

        # if attention_mask is None:
        #     loss = self.loss(model_out, query_noise)
        # else:
        #     loss = self.loss(model_out, query_noise)
        #     loss_mask = repeat(attention_mask, "b s -> b s d", d=3)
        #     loss *= loss_mask
        # loss = reduce(loss, "b ... -> b (...)", "mean")

        # return loss.mean()

        loss = self.loss(model_out, query_noise)
        loss_mask = repeat(attention_mask, "b s -> b s d", d=3)
        loss *= loss_mask
        
        loss = reduce(loss, "b ... -> b", "sum")
        loss_mask = reduce(loss_mask, "b ... -> b", "sum")
        loss = loss / loss_mask
        loss = loss.mean()
        return loss

    def forward(self, batch):
        x = batch[:, :, : -self.input_signal_num_channels]
        y = batch[:, :, -self.input_signal_num_channels :]

        y_shape = y.shape
        y_noise = torch.randn(y_shape, device=self.device)

        shape = y_noise.shape
        y_sampled = self.sampling(
            shape,
            x,
            y_noise,
        )

        return y_sampled

    def training_step(self, batch, batch_idx, noise=None):
        batch, attention_mask, mols, _, _, _, _ = batch
        label = None

        x = batch[:, :, : -self.input_signal_num_channels]
        y = batch[:, :, -self.input_signal_num_channels :]
        noise = torch.randn_like(y)

        # Select random context coordinate positions
        # npoints_context = self.npoints_context[0]
        # context_indices = torch.randperm(x.shape[1])[:npoints_context]

        # not applying randperm when using attention_mask
        npoints_context = min((self.npoints_context[0], x.shape[1]))
        if self.randperm and attention_mask is None:
            context_indices = torch.randperm(x.shape[1])[:npoints_context]
        else:
            context_indices = torch.arange(npoints_context)

        context_x = x[:, context_indices]
        context_x = context_x.to(self.device)
        context_y = y[:, context_indices]
        context_noise = noise[:, context_indices]

        # Select random query coordinate positions (may or may not intersect with context)
        # npoints_query = self.npoints_query[0]
        # query_indices = torch.randperm(x.shape[1])[:npoints_query]
        
        # not applying randperm when using attention_mask
        npoints_query = min((self.npoints_query[0], x.shape[1]))
        if self.randperm and attention_mask is None:
            query_indices = torch.randperm(x.shape[1])[:npoints_query]
        else:
            query_indices = torch.arange(npoints_query)

        query_x = x[:, query_indices]
        query_y = y[:, query_indices]
        query_noise = noise[:, query_indices]

        t = torch.randint(0, self.num_timesteps, (y.shape[0],)).to(self.device)

        loss = self.p_losses(
            context_x,
            context_y,
            t,
            query_x,
            query_y,
            context_noise,
            query_noise,
            label=label,
            attention_mask=attention_mask,
        )

        self.log(
            "loss/mse",
            loss.item(),
            on_epoch=True,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if not self.online_sample:
            return 
        # batch, attention_mask, mols, normalizer = batch
        (
            batch,
            attention_mask,
            mols,
            normalizer,
            smiles,
            conf_indices,
            num_atoms,
        ) = batch
        label = None

        x = batch[:, :, : -self.input_signal_num_channels]
        y = batch[:, :, -self.input_signal_num_channels :]

        # for molecule with K conformers, we sample 2K conformers
        n_mols = len(mols)
        x_batch = torch.cat([x, x], dim=0)
        attention_mask_batch = torch.cat([attention_mask, attention_mask], dim=0)
        shape = [x_batch.shape[0], x_batch.shape[1], 3]
        y_noise = torch.randn(shape, device=self.device)

        pos_sampled = self.sampling(
            shape,
            x_batch,
            y_noise,
            attention_mask=attention_mask_batch,
        )

        # rescale the sampled coords to original scale
        # [-1, 1] -> [0, 1]
        pos_sampled = (pos_sampled + 1) * 0.5

        # [0, 1] -> original scale
        min_x, max_x, min_y, max_y, min_z, max_z = normalizer
        pos_sampled[..., 0] = pos_sampled[..., 0] * (max_x - min_x) + min_x
        pos_sampled[..., 1] = pos_sampled[..., 1] * (max_y - min_y) + min_y
        pos_sampled[..., 2] = pos_sampled[..., 2] * (max_z - min_z) + min_z

        for i in range(len(smiles)):
            smi = smiles[i]
            mol = mols[i]

            pos1 = pos_sampled[i][: num_atoms[i]]
            mol_sampled1 = set_rdmol_positions(mol, pos1)
            if smi not in self.sample_mols_dict:
                self.sample_mols_dict[smi] = []
            if smi not in self.gt_mols_dict:
                self.gt_mols_dict[smi] = []
            self.sample_mols_dict[smi].append(mol_sampled1)
            self.gt_mols_dict[smi].append(mol)

            pos2 = pos_sampled[i + n_mols][: num_atoms[i]]
            mol_sampled2 = set_rdmol_positions(mol, pos2)
            self.sample_mols_dict[smi].append(mol_sampled2)

        return

    def on_validation_epoch_start(self):
        self.gt_mols_dict = {}
        self.sample_mols_dict = {}
        return

    def on_validation_epoch_end(self):
        # all_gather_object doesn't support async_op=True
        # gather results on all processes to avoid deadlock
        world_size = tdist.get_world_size()
        gt_mols_dict_list = [None for _ in range(world_size)]
        tdist.all_gather_object(gt_mols_dict_list, self.gt_mols_dict)
        sample_mols_dict_list = [None for _ in range(world_size)]
        tdist.all_gather_object(sample_mols_dict_list, self.sample_mols_dict)

        # only save and evaluate samples on rank 0
        if tdist.get_rank() != 0:
            del gt_mols_dict_list, sample_mols_dict_list
            return

        gt_mols_dict_all, sample_mols_dict_all = {}, {}
        for gt_mols_dict, sample_mols_dict in zip(
            gt_mols_dict_list, sample_mols_dict_list
        ):
            for smi, gts in gt_mols_dict.items():
                if smi not in gt_mols_dict_all:
                    gt_mols_dict_all[smi] = []
                gt_mols_dict_all[smi].extend(gts)
            for smi, samples in sample_mols_dict.items():
                if smi not in sample_mols_dict_all:
                    sample_mols_dict_all[smi] = []
                sample_mols_dict_all[smi].extend(samples)

        gt_mols_list = []
        sample_mols_list = []
        smiles_list = []
        for smi, gts in gt_mols_dict_all.items():
            smiles_list.append(smi)
            gt_mols_list.append(gts)
            sample_mols_list.append(sample_mols_dict_all[smi])

        conformer_dict = {
            "smiles": smiles_list,
            "ground_truth": gt_mols_list,
            "model_samples": sample_mols_list,
        }
        save_path = os.path.join(
            self.viz_dir, f"samples_epoch_{self.current_epoch}.pkl"
        )
        with open(save_path, "wb") as f:
            pickle.dump(conformer_dict, f)
        print("saved samples to {}".format(save_path))

        # evaluate molecular conformer samples only when requested (can take a long time on DRUGS/XL)
        if self.online_evaluation and tdist.get_rank() == 0:
            (
                rmsd_conf,
                covr_scores,
                matr_scores,
                covp_scores,
                matp_scores,
            ) = self.metrics_molecule(sample_mols_list, gt_mols_list)

            metrics_dict = {
                "rmsd": rmsd_conf,
                "covr": covr_scores,
                "matr": matr_scores,
                "covp": covp_scores,
                "matp": matp_scores,
            }
            save_path = os.path.join(
                self.viz_dir, f"metrics_epoch_{self.current_epoch}.pkl"
            )
            with open(save_path, "wb") as f:
                pickle.dump(metrics_dict, f)
            print("saved metrics to {}".format(save_path))

            covr_mean = np.mean(covr_scores)
            covr_median = np.median(covr_scores)
            matr_mean = np.nanmean(matr_scores)
            matr_median = np.nanmedian(matr_scores)
            covp_mean = np.mean(covp_scores)
            covp_median = np.median(covp_scores)
            matp_mean = np.nanmean(matp_scores)
            matp_median = np.nanmedian(matp_scores)
            print("covr-mean:", covr_mean * 100, "covr-median:", covr_median * 100)
            print("matr-mean:", matr_mean, "matr-median:", matr_median)
            print("covp-mean:", covp_mean * 100, "covp-median:", covp_median * 100)
            print("matp-mean:", matp_mean, "matp-median:", matp_median)

            self.log(
                "metrics/covr-mean",
                covr_mean,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                rank_zero_only=True,
                sync_dist=True,
            )
            self.log(
                "metrics/covr-median",
                covr_median,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                rank_zero_only=True,
                sync_dist=True,
            )
            self.log(
                "metrics/matr-mean",
                matr_mean,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                rank_zero_only=True,
                sync_dist=True,
            )
            self.log(
                "metrics/matr-median",
                matr_median,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                rank_zero_only=True,
                sync_dist=True,
            )
            self.log(
                "metrics/covp-mean",
                covp_mean,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                rank_zero_only=True,
                sync_dist=True,
            )
            self.log(
                "metrics/covp-median",
                covp_median,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                rank_zero_only=True,
                sync_dist=True,
            )
            self.log(
                "metrics/matp-mean",
                matp_mean,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                rank_zero_only=True,
                sync_dist=True,
            )
            self.log(
                "metrics/matp-median",
                matp_median,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                rank_zero_only=True,
                sync_dist=True,
            )

            self.log(
                "metrics/checkpoint_metric",
                matp_mean,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                rank_zero_only=True,
                sync_dist=True,
            )

        gc.collect()

    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        self.log(
            "hparams/lr",
            # scheduler.get_last_lr()[0],
            # optimizer.param_groups[0]['lr'], 
            scheduler.get_lr()[0],
            on_epoch=True,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
        )

    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.model_ema:
            self.model_ema.update(self.model)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config["lr"],
        )

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.train_config["max_steps"],
            max_lr=self.train_config["lr"],
            min_lr=0.0,
            warmup_steps=self.train_config["warmup_steps"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint):
        # save the config if its available
        try:
            checkpoint["opt"] = self.opt
        except Exception:
            pass
