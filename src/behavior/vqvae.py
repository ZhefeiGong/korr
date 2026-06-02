from typing import List, Optional, Tuple, Union
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor
from src.dataset.normalizer import LinearNormalizer
from src.carp.optim.amp_opt import AmpOptimizer
from src.carp.svqvae import VQVAE, VectorQuantizer2

class ActionVQVAETrainer(object):
    def __init__(
        self, 
        device, 
        normalizer: LinearNormalizer,
        vae_wo_ddp: VQVAE,
        vae_opt: AmpOptimizer,
        ema_ratio: float,
        is_ema: bool,
        act_dim_sep: int,
        act_dim_names: List[str],
    ):  
        
        super(ActionVQVAETrainer, self).__init__()
        
        ### models - vae
        self.vae_opt = vae_opt
        self.vae_wo_ddp: VQVAE = vae_wo_ddp
        self.vae_params: Tuple[nn.Parameter] = tuple(self.vae_wo_ddp.parameters())
        self.act_dim_sep = act_dim_sep
        
        ### ema for vae
        self.ema_ratio = ema_ratio
        self.is_ema = is_ema
        if self.is_ema:
            self.vae_ema: VQVAE = deepcopy(vae_wo_ddp).eval()
        else:
            self.vae_ema: VQVAE = None
        
        ### normalizer
        self.vae_norm = normalizer
        self.vae_norm.to(device)
        
        ### params - vae
        self.w_l1=0.0 # discarded
        self.w_l2=1.0 # 
        self.w_vq=1.0 # 0.25(smaller) | 1.00(normal)
        self.w_lp=0.0 # discarded
        
        ### usage name for recording
        self.usage_names = act_dim_names
        self.is_loggable = True

    def train(self):
        """
        @func: 
        transfer vqvae into train mode
        """
        self.vae_wo_ddp.train()
        return    
    
    def eval(self):
        """
        @func: 
        transfer vqvae into evaludate mode
        """
        self.vae_wo_ddp.eval()
        return    
    
    def compute_loss(self, batch):
        """
        @func: 
        compute the training loss
        """

        # automatic mixed precision
        with self.vae_opt.amp_ctx:    
            # data initialization
            inp = batch['action']
            inp = inp.view(inp.shape[0],1,inp.shape[1],inp.shape[2]).contiguous()
            # forward
            self.vae_wo_ddp.forward
            inp_slice = inp[:,:,:,self.act_dim_sep:self.act_dim_sep+1]
            rec_inp_slice, usages, loss_vq = self.vae_wo_ddp(inp=inp_slice, ret_usages=self.is_loggable)
            # calc loss
            loss_rec_l1 = F.l1_loss(input=rec_inp_slice, target=inp_slice)
            loss_rec_l2 = F.mse_loss(input=rec_inp_slice, target=inp_slice)
            # combine loss
            loss_vae = self.w_l2 * loss_rec_l2 + self.w_vq * loss_vq

        # loss record
        losses_log = {
            'l1_loss': loss_rec_l1.item(),
            'l2_loss': loss_rec_l2.item(),
            'vq_loss': loss_vq.item(),
        }
        
        # ### VAE Backward
        # grad_norm, scale_log2 = self.vae_opt.backward_clip_step(loss=loss_vae, stepping=True)
        
        # ### UPDATE | EMA
        # if self.is_ema:
        #     self.ema_update(g_it)
        
        # ### LOG to metric
        # if it == 0 or it in me_lg.log_iters:
        #     me_lg.update(
        #         loss_rec_l1=loss_rec_l1.item(), 
        #         loss_rec_l2=loss_rec_l2.item(), 
        #         loss_vq=loss_vq.item(), 
        #         loss_vae=loss_vae.item())
        
        # ### LOG to tensorboard
        # if is_loggable:
        #     # loss
        #     tb_lg.update(
        #         head='VAE_iter_loss',
        #         loss_rec_l1=loss_rec_l1.item(), 
        #         loss_rec_l2=loss_rec_l2.item(), 
        #         loss_vq=loss_vq.item(), 
        #         loss_vae=loss_vae.item(),
        #         step=g_it)
        #     # usage
        #     name = self.usage_names[self.act_dim_sep]
        #     tb_lg.update(head=f"VAE_vocab_usage_{name}",
        #                 scale_1=usages[0],
        #                 scale_2=usages[1],
        #                 scale_3=usages[2],
        #                 scale_4=usages[3],
        #                 scale_all=usages[-1],
        #                 step=g_it)
        
        return loss_vae, losses_log, usages
    
    def backward_update(self, loss, g_it):
        """
        @func: 
        ema update and backward update
        """

        # vqvae backward
        grad_norm, scale_log2 = self.vae_opt.backward_clip_step(loss=loss, stepping=True)

        # update for ema
        if self.is_ema:
            self.ema_update(g_it)
        
        return grad_norm, scale_log2
    
    def ema_update(self, g_it):
        """
        @func: 
        ema update in order to get a more stable version
        """
        ## init
        ema_ratio = min(self.ema_ratio, (g_it//2 + 1) / (g_it//2 + 10))
        ## params
        for p_ema, p in zip(self.vae_ema.parameters(), self.vae_wo_ddp.parameters()):
            if p.requires_grad:
                p_ema.data.mul_(ema_ratio).add_(p.data, alpha=1-ema_ratio)
        ## buffer
        for p_ema, p in zip(self.vae_ema.buffers(), self.vae_wo_ddp.buffers()):
            p_ema.data.copy_(p.data)
        ## codebook - update Z through `grad` with default, so no need for the following update during ema
        quant, quant_ema = self.vae_wo_ddp.quantizer, self.vae_ema.quantizer
        quant: VectorQuantizer2
        if hasattr(quant, 'using_ema') and quant.using_ema: # then embedding.weight requires no grad, thus is not in self.vae_ema.parameters(); so need to update it manually
            if hasattr(quant, 'using_restart') and quant.using_restart: # cannot use ema, cuz quantize.embedding uses replacement (random restart)
                quant_ema.embedding.weight.data.copy_(quant.embedding.weight.data) 
            else:
                quant_ema.embedding.weight.data.mul_(ema_ratio).add_(quant.embedding.weight.data, alpha=1-ema_ratio)
    
    def get_config(self):
        """
        @func: 
        get the loss and ema config of the model
        """
        return {
            'ema_ratio': self.ema_ratio,
            'w_l1': self.w_l1,
            'w_l2': self.w_l2, 
            'w_vq': self.w_vq, 
            'w_lp': self.w_lp,
        }
    
    def state_dict(self):
        """
        @func: 
        fetch the models needed to reserve
        """
        state = {'config': self.get_config()}
        for k in ('vae_wo_ddp', 'vae_ema', 'vae_opt', 'vae_norm'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True):
        """
        @func: 
        load the models needed
        """
        for k in ('vae_wo_ddp', 'vae_ema', 'vae_opt', 'vae_norm'):
            m = getattr(self, k)
            if m is not None:
                
                # model
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                
                # load_state_dict
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VAETr.load_state_dict] {k} missing:  {missing}')
                    print(f'[VAETr.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAETr.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)



