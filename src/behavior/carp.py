from collections import deque
from src.common.geometry import proprioceptive_quat_to_6d_rotation
from omegaconf import DictConfig
from src.models.vision import DualInputAttentionPool2d
import torch
import torch.nn as nn

from src.dataset.normalizer import LinearNormalizer
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src.behavior.base import Actor
from src.behavior.ema import EMAModel
from src.models import get_diffusion_backbone

from ipdb import set_trace as bp  # noqa
from typing import Tuple, Union

import copy
from functools import partial
from src.carp.MSAT.vqvae import MultiScaleActionTokenizer
from src.carp.CFAP.autoreg import Coarse2FineAutoRegressor
from src.carp.optim.amp_opt import AmpOptimizer
from src.carp.CFAP import build_vae_ar
from src.carp.optim.lr_control import filter_params
from omegaconf import DictConfig, OmegaConf

def load_sep_vae_model(vae_local, vae_ckpt_paths):
    """
    @func:
    load vae model separately | [x + y + z + rotation6d + gripper]
    """
    for idx in range(len(vae_ckpt_paths)):
        # load corresponding config
        cfg: DictConfig = OmegaConf.create(torch.load(vae_ckpt_paths[idx])["config"])
        # cosine | euler | no need for strict matching
        vae_local.load_state_dict_sep(torch.load(vae_ckpt_paths[idx], map_location='cpu')['model_state_dict']['vae_wo_ddp'], 
                                      act_dim=idx, 
                                      strict=False, 
                                      using_znorm = cfg.vqvae.vqnorm)
        print(f'[INFO] load vae ckpt {vae_ckpt_paths[idx]}')
    return vae_local

class Coarse2FineAutoRegressivePolicy(Actor):

    def __init__(
        self,
        device: Union[str, torch.device],
        cfg: DictConfig,
    ) -> None:
        
        super().__init__(device, cfg)
        actor_cfg = cfg.actor
        ### dimension of the action
        self.action_dim = actor_cfg.act_dim
        ### models
        self.vae_local, self.ar_wo_ddp = self._init_models(actor_cfg, device)
        self.ar_opt = self._init_opt(actor_cfg, self.ar_wo_ddp)
        self.ema_ar_wo_ddp: Coarse2FineAutoRegressor = copy.deepcopy(self.ar_wo_ddp)
        self.ema_func = EMAModel(inv_gamma=1.0, 
                                 max_value=0.9999, 
                                 min_value=0.0, 
                                 power=0.75,
                                 update_after_step=0,
                                 model = self.ema_ar_wo_ddp)
        ### loss
        self.patch_nums, self.resos = [x * self.action_dim for x in actor_cfg.patch_nums], actor_cfg.resos
        self.label_smooth = actor_cfg.tls # whetehr smooth the gt labels
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=actor_cfg.tls, reduction='none') # reduction -> untouch value | [0.5, 1.0, 1.5]
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean') # reduction -> mean value | (0.5 + 1.0 + 1.5) / 3 = 1.0
        self.L = sum(pn * 1 for pn in self.patch_nums) # length for all of the scale 
        self.last_l = self.patch_nums[-1] * 1 # the length of the last scale
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L # for training loss
        ### get the section for each scale
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn * 1))
            cur += pn*1
        ### others
        self.normalizer = LinearNormalizer()
        self.ar_norm = self.normalizer
        self.eta = 0.0

    def _init_models(self, actor_cfg, device):
        """
        @func:
        initialize the models of carp
        """
        ### load vae and ar
        vae_local, ar_wo_ddp = build_vae_ar(
            device=device,
            patch_nums=actor_cfg.patch_nums,
            ## multi-scale action tokenization
            V=actor_cfg.vocab_size, 
            Cvae=actor_cfg.vocab_ch, 
            ch=actor_cfg.vch, 
            action_dim=actor_cfg.act_dim,
            num_actions=actor_cfg.act_horizon,
            dropout=actor_cfg.vdrop,
            beta=actor_cfg.vqbeta,
            using_znorm=actor_cfg.vqnorm,
            quant_conv_ks=3, # fixed
            quant_resi=actor_cfg.vqresi,
            share_quant_resi=4, # fixed
            ## coarse-to-fine autoregressive prediction
            obs_encoder = None, 
            depth=actor_cfg.tdepth, 
            n_obs_steps=actor_cfg.tnobs, 
            obs_dim=actor_cfg.obs_dim,
            embed_dim=actor_cfg.tembed,
            shared_aln=actor_cfg.saln, # whether to use shared adaln
            attn_l2_norm=actor_cfg.anorm, # whether to use L2 normalized attention
            init_adaln=actor_cfg.taln, # for coarse-to-fine autoregressive prediction
            init_adaln_gamma=actor_cfg.talng, # for coarse-to-fine autoregressive prediction
            init_head=actor_cfg.thd, # for coarse-to-fine autoregressive prediction
            init_std=actor_cfg.tini, # for coarse-to-fine autoregressive prediction
        )
        ### load the model of vae
        if hasattr(vae_local, '_orig_mod'):
            vae_local = vae_local._orig_mod # from ddp to original model
        vae_local = load_sep_vae_model(vae_local, actor_cfg.vae_ckpt_paths)
        ### set to required-gradient or non-required-gradient
        vae_local: MultiScaleActionTokenizer = vae_local
        ar_wo_ddp: Coarse2FineAutoRegressor = ar_wo_ddp
        assert all(p.requires_grad is False for p in vae_local.parameters())
        assert all(p.requires_grad is True for p in ar_wo_ddp.parameters())
        ### showcase
        print(f'[INIT] AR model = {ar_wo_ddp}\n\n')
        count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.4f}M'
        print(f'[INIT][#para] ' + ', '.join([
            f'{k}={count_p(m) if m is not None else "0"}' for k, m in (('VAE', vae_local), 
                                                                       ('VAE.enc', vae_local.encoders), 
                                                                       ('VAE.dec', vae_local.decoders), 
                                                                       ('VAE.quant', vae_local.quantizers))]))
        print('[INIT][#para] ' + ', '.join([
            f'{k}={count_p(m) if m is not None else "0"}' for k, m in [('AR', ar_wo_ddp), 
                                                                    ('AR.enc', ar_wo_ddp.obs_encoder)]]))
        ### return
        return vae_local, ar_wo_ddp

    def _init_opt(self, actor_cfg, ar_wo_ddp):
        """
        @func:
        initialize the optimizer of carp
        """
        ### construct the params for building optimizer
        names, paras, para_groups = filter_params(ar_wo_ddp, nowd_keys={
            ###
            'cls_token', 
            'start_token', 
            'task_token', 
            'cfg_uncond',
            'pos_embed', 
            'pos_1LC', 
            'pos_start', 
            'start_pos', 
            'lvl_embed',
            'gamma', 
            'beta',
            'ada_gss', 
            'moe_bias',
            'scale_mul',
            ### 
            'class_emb', 
            'embedding',
            'norm_scale',
        })
        opt_clz = {
            'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=actor_cfg.afuse),
            'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=actor_cfg.afuse),
        }[actor_cfg.opt.lower().strip()]
        opt_kw = dict(lr=actor_cfg.tlr, weight_decay=0)
        print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')
        ### build the optimizer
        ar_opt = AmpOptimizer(
            mixed_precision=actor_cfg.fp16, 
            optimizer=opt_clz(params=para_groups, **opt_kw), 
            names=names,
            paras=paras,
            grad_clip=actor_cfg.tclip, # whether to utilize the gradient clipping
            n_gradient_accumulation=actor_cfg.ac
        )
        ### return
        return ar_opt
    
    def trans_sep_idxBls(self, idxBls):
        """
        @func: 
        transform the separate idxBls into the format of the input of ar
        """
        B = idxBls[0][0].shape[0]
        gt_BL = torch.cat([torch.cat(idxBl, dim=1).unsqueeze(1) for idxBl in idxBls], dim=1) # [B,action_dim,10]
        gt_BL = torch.transpose(gt_BL, 1, 2) # shape->[B, 10, action_dim]
        gt_BL = gt_BL.reshape(B,-1) # shape->[B, 10*action_dim]
        return gt_BL
    
    def trans_sep_ar_inputs(self, ar_inputs):
        """
        @func: 
        transform the separate ar_input into the format of the input of ar
        """
        B,_,C = ar_inputs[0].shape
        input = torch.cat([x.unsqueeze(1) for x in ar_inputs], dim=1) # [B,action_dim,9,C]
        input = torch.transpose(input, 1, 2) # shape->[B,9,action_dim,C]
        input = input.reshape(B,-1, C) # shape->[B,9*action_dim,C]
        return input
    
    # === Inference ===
    def _normalized_action(self, nobs: torch.Tensor) -> torch.Tensor:
        """
        @func:
        inference / test the action prediction process
        """
        # Get batch size
        B = nobs.shape[0]
        if len(nobs.shape) == 2:
            nobs = nobs.reshape(B, self.obs_horizon, self.timestep_obs_dim)
        # Predict the actions -> utilize ema here
        naction = self.ema_ar_wo_ddp.autoregressive_infer_cfg(nobs=nobs, vae_proxy=self.vae_local) # B1LC -> normalized actions here
        naction = naction.squeeze(1) # B1LC -> BLC
        # Return
        return naction
    
    # === Training ===
    def compute_loss(self, batch):
        """
        @func:
        compute the loss during the trianing process
        """

        ### Get data
        # State already normalized in the dataset
        obs_cond = self._training_obs(batch, flatten=self.flatten_obs)
        if len(obs_cond.shape) == 2:
            B = obs_cond.shape[0]
            obs_cond = obs_cond.reshape(B, self.obs_horizon, self.timestep_obs_dim) # shape: [B, obs_horizon, timestep_obs_dim]
        # Action already normalized in the dataset
        # These actions are the exact ones we should predict, i.e., the handling of predicting past actions or not is also handled in the dataset class
        naction = batch["action"]  # shape: [B, L, action_dim]
        naction = naction.unsqueeze(1).contiguous()  # shape: [B, 1, L, action_dim]

        ### Data preparation
        B, V = naction.shape[0], self.vae_local.vocab_size # batch_size, vocab_size
        idxBls = self.vae_local.inp_to_idxBl(naction) # List[List([B,1or2or3or4])]
        ar_inputs = self.vae_local.idxBl_to_autoreg_input(idxBls) # List[[B,2+3+4,c])]
        gt_BL = self.trans_sep_idxBls(idxBls) # [B，10*action_dim]
        x_BLCv_wo_first_l = self.trans_sep_ar_inputs(ar_inputs) # [B, 9*action_dim, 8]

        ### Train
        with self.ar_opt.amp_ctx:
            ## forward
            self.ar_wo_ddp.forward # forward
            logits_BLV = self.ar_wo_ddp(obs_cond, x_BLCv_wo_first_l) # transformer inference | BLV | multi-task
            ## loss
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1) # calc the corss-entropy loss |  [B,L,V] and [B,L] | get [B,L]
            # final loss
            lw = self.loss_weight # 1L
            loss = loss.mul(lw).sum(dim=-1).mean() # get [B,]
        losses_log = {"bc_loss": loss.item()}

        ### Record
        # the rate of vocabularies we often use
        pred_BL = logits_BLV.data.argmax(dim=-1)
        prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
        prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
        cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
        kw = dict(z_voc_usage=cluster_usage)
        # accuracy and loss of each layer
        for si, (bg, ed) in enumerate(self.begin_ends):
            pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1) # get ans
            acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100 # the accuracy between predicted idx and target idx
            ce = self.val_loss(pred, tar).item() # cross-entropy loss between [L,V] and [L,]
            kw[f'acc_{self.resos[si]}'] = acc # 
            kw[f'L_{self.resos[si]}'] = ce  #

        # # Rescale for different domains for image-based -> sim & real
        # if self.rescale_loss_for_domain:
        #     # Calculate class weights
        #     class_sizes = torch.bincount(batch["domain"].squeeze())
        #     class_weights = torch.pow(class_sizes.float(), -1.0 / 2)
        #     class_weights = class_weights / class_weights.sum()
        #     # Apply class weights to the loss
        #     class_weights = class_weights[batch["domain"]]
        #     loss *= class_weights

        # # Add the VIB loss for image-based -> sim & real
        # if self.camera_2_vib is not None:
        #     mu, log_var = batch["mu"], batch["log_var"]
        #     vib_loss = self.camera_2_vib.kl_divergence(mu, log_var)
        #     # Clip the VIB loss to prevent it from dominating the total loss
        #     losses["vib_loss"] = vib_loss.item()
        #     vib_loss = torch.clamp(vib_loss, max=1)
        #     # Scale the VIB loss by the beta and add it to the total loss
        #     loss += self.vib_front_feature_beta * vib_loss
        # # Add the confusion loss for image-based -> sim & real
        # if self.confusion_loss_beta > 0:
        #     confusion_loss = batch["confusion_loss"]
        #     losses["confusion_loss"] = confusion_loss.item()
        #     loss += self.confusion_loss_beta * confusion_loss

        return loss, losses_log, kw
    
    # === Training ===
    def backward_update(self, loss):
        """
        @func: 
        ema update and backward update
        """

        # vqvae backward
        grad_norm, scale_log2 = self.ar_opt.backward_clip_step(loss=loss, stepping=True)

        # update for ema
        self.ema_func.step(self.ar_wo_ddp)
        
        return grad_norm, scale_log2
    
    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 
            'resos': self.resos,
            'label_smooth': self.label_smooth,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('ar_wo_ddp', 'vae_local', 'ar_opt', 'ar_norm', 'ema_ar_wo_ddp'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod # from ddp to original model
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('ar_wo_ddp', 'vae_local', 'ar_opt', 'ar_norm', 'ema_ar_wo_ddp'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod # from ddp to original model
                
                if isinstance(m, LinearNormalizer):
                    m.load_state_dict(state[k])
                else:
                    ret = m.load_state_dict(state[k], strict=strict)
                    if ret is not None:
                        missing, unexpected = ret
                        print(f'[ARTrainer.load_state_dict] {k} missing:  {missing}')
                        print(f'[ARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[AR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)


