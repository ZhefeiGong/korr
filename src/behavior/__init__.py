from omegaconf import DictConfig
from src.behavior.base import Actor

def get_actor(cfg: DictConfig, device) -> Actor:
    """Returns an actor model."""
    actor_name = cfg.actor_name if "actor_name" in cfg else cfg.actor.name
    obs_type = cfg.observation_type

    assert obs_type in ["image", "state"], f"Invalid observation type: {obs_type}"

    # if actor_name == "mlp":
    #     from src.behavior.mlp import MLPActor
    #     return MLPActor(
    #         cfg=cfg,
    #         device=device,
    #     )
    # elif actor_name == "diffusion":
    
    if actor_name == "diffusion":
        from src.behavior.diffusion import DiffusionPolicy # NOTE: legacy diffusion policy
        return DiffusionPolicy(
            cfg=cfg,
            device=device,
        )
    
    elif actor_name == "residual_diffusion":
        from src.behavior.residual_diffusion import ResidualDiffusionPolicy # NOTE: legacy diffusion policy plus `residual policy`
        return ResidualDiffusionPolicy(
            cfg=cfg,
            device=device,
        )
    
    # elif actor_name == "attentionpool_diffusion":
    #     from src.behavior.diffusion import AttentionPoolDiffusionPolicy # NOTE: only for vision-based diffusion policy
    #     return AttentionPoolDiffusionPolicy(
    #         cfg=cfg,
    #         device=device,
    #     )
    
    elif actor_name == "carp": 
        from src.behavior.carp import Coarse2FineAutoRegressivePolicy # NOTE: coarse-to-fine autoregressive policy
        return Coarse2FineAutoRegressivePolicy(
            cfg=cfg,
            device=device
        )
    
    elif actor_name == 'residual_carp':
        from src.behavior.residual_carp import ResidualCARP # NOTE: coarse-to-fine autoregressive policy plus `residual policy`
        return ResidualCARP(
            cfg=cfg,
            device=device
        )
    
    
    raise ValueError(f"Unknown actor type: {cfg.actor}")
