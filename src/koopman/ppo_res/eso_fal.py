"""
@author: 
| copyright @ Zhefei(Jeffrey) Gong
@date: 
| Apr.8th 2025
@func: 
| the core implementation of Extended State Observer
@link: 
| ARDC: From PID to Active Disturbance Rejection Control
| RAL: Koopman-based Robust Learning Control with Extended State Observer
"""

import torch
import torch.nn as nn

class ExtendedStateObserver:
    def __init__(self, 
                 n_envs=1024,
                 g_affine=None,
                 u_affine=None,
                 lift_transform=None,
                 device='cpu',
                 delta_t=0.1, # 0.001 -> 0.1
                 beta=[10.0, 20.0, 40.0], # [200,1200,2500]
                 alpha=[0.5, 0.25], # [0.5, 0.25] 
                 delta=0.15, # 0.01
                 ): 
        """
        KoopmanESO implements an Extended State Observer for Koopman-linearized systems.

        Args:
            g_affine (tensor): Koopman A matrix.
            u_affine (tensor): Koopman B matrix
            lift_transform (nn.module): Lifted function in Koopman
        """
        super().__init__()

        # Koopman
        assert (g_affine is not None) and (u_affine is not None), "receive the g_affine or u_affine with value of None"
        self.g_affine = torch.Tensor(g_affine).to(device) # Koopman A
        self.u_affine = torch.Tensor(u_affine).to(device) # Koopman B
        self.u_affine_pinv = torch.linalg.pinv(self.u_affine) # inverse of B
        self.g_dim = self.g_affine.shape[-1] # dimension of state
        self.u_dim = self.u_affine.shape[-1] # dimension of action
        self.lift_transform = lift_transform

        # Extended State Observer
        self.beta1, self.beta2, self.beta3 = beta
        self.alpha1, self.alpha2 = alpha
        self.delta = delta

        # Others
        self.delta_t = delta_t
        self.device = device
        self.n_envs = n_envs

        # Initialize observer states: estimated z1, z2, z3
        self.z1_hat = torch.zeros((n_envs, self.g_dim)).to(device)
        self.z2_hat = torch.zeros((n_envs, self.g_dim)).to(device)
        self.z3_hat = torch.zeros((n_envs, self.g_dim)).to(device)

        # Build d_hat
        self.d_hat = torch.zeros((n_envs, self.u_dim)).to(device)

        # Storage
        self.error_storage = []
    
    def reset(self):
        """Reset the observer states to zero."""
        self.z1_hat.zero_()
        self.z2_hat.zero_()
        self.z3_hat.zero_()
    
    def fal(self, e, delta, alpha):
        """Nonlinear error function used in ESO."""
        return torch.where(
            torch.abs(e) <= delta,
            e / (delta ** (1 - alpha)),
            torch.abs(e) ** alpha * torch.sign(e)
        )
    
    def get_d_hat(self):
        """Get the correlation from the last step"""
        return self.d_hat
    
    def set_z1_hat(self, z1_hat_raw):
        """Set the value of z1_hat"""
        assert z1_hat_raw.shape[0] == self.n_envs, f"Expected shape ({self.n_envs}, ...), but got {z1_hat_raw.shape}"
        self.z1_hat = self.lift_transform(z1_hat_raw)
        return
    
    def predict_eso(self, z1_true_raw, z1_true_before_raw, u):
        """
        Perform one step of ESO update.

        Args:
            z1_true_raw (Tensor): Current true observation z1(k), shape [N, g_dim]
            u (Tensor): Current control input u(k), shape [N, u_dim]
        Returns:
            Tuple[Tensor, Tensor, Tensor]: estimated z1_hat, z2_hat, z3_hat
        """

        ### Lift raw z1_true
        z1_true = self.lift_transform(z1_true_raw) # [N,g_dim]
        z1_true_before = self.lift_transform(z1_true_before_raw) # [N,g_dim]

        # ### from continuous to discrete system
        # ### Observation error
        # e1 = z1_true - self.z1_hat # [N,g_dim]
        # e1 = self.z1_hat - z1_true # [N,g_dim]
        # self.error_storage.append(e1)
        # print("error : ", e1[0][0])
        # ### Run here
        # z1_hat_next = self.z1_hat + self.delta_t * (self.z2_hat - self.beta1 * e1) # [N,g_dim]
        # # z2_hat_next = self.z2_hat + self.delta_t * (
        # #     self.z3_hat + self.predict_kpm(z1_true_before, u) - self.beta2 * self.fal(e1, self.delta, self.alpha1)
        # # ) # [N,g_dim]
        # z2_hat_next = self.z2_hat + self.delta_t * (
        #     self.z3_hat + self.predict_kpm(self.z2_hat, u) - self.beta2 * self.fal(e1, self.delta, self.alpha1)
        # ) # [N,g_dim]
        # z3_hat_next = self.z3_hat + self.delta_t * (-self.beta3 * self.fal(e1, self.delta, self.alpha2)) # [N,g_dim]
        # ### Save the new states
        # self.z1_hat = z1_hat_next
        # self.z2_hat = z2_hat_next
        # self.z3_hat = z3_hat_next
        # ### Calculate the correlation of u
        # self.d_hat = torch.matmul(self.z2_hat, self.u_affine_pinv.transpose(0, 1)) # [N,u_dim]
        # ### Return \hat{d}
        # return self.z1_hat, self.z2_hat, self.z2_hat

        ### specific discrete system
        e1 = self.z1_hat - z1_true # [N,g_dim]
        self.error_storage.append(e1)
        print("error : ", e1[0][0])
        z1_hat_next = self.z2_hat + self.predict_kpm(self.z1_hat, u) - self.beta2 * self.fal(e1, self.delta, self.alpha1) # [N,g_dim]
        z2_hat_next = self.z2_hat - self.beta3 * self.fal(e1, self.delta, self.alpha2)# [N,g_dim]
        self.z1_hat = z1_hat_next
        self.z2_hat = z2_hat_next
        self.d_hat = torch.matmul(self.z2_hat, self.u_affine_pinv.transpose(0, 1)) # [N,u_dim]
        return self.z1_hat, self.z2_hat, self.z2_hat

    def predict_kpm(self, G, U):
        '''
        Intro:
            x_{t+1} = Ax_t + Bu
        Func:
            predict dynamics with current koopman parameters
            note both input and return are embeddings of the predicted state, we can recover that by using invertible net, e.g. normalizing-flow models
            but that would require a same dimensionality
        '''
        return torch.matmul(G, self.g_affine.transpose(0, 1)) + torch.matmul(U, self.u_affine.transpose(0, 1))

    def save_error_storage(self, path):
        save_storage = torch.stack(self.error_storage, dim=1)
        torch.save(save_storage, path)
        return

