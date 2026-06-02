"""
@author: 
| copyright @ Zhefei(Jeffrey) Gong
@date: 
| Apr.10th 2025
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

        # Others
        self.device = device
        self.n_envs = n_envs

        # Initialize observer states: estimated z1 and d_hat
        self.z1_hat = torch.zeros((n_envs, self.g_dim)).to(device)
        self.d_hat = torch.zeros((n_envs, self.g_dim)).to(device)

        # Initialize extented parameters
        self.id_affine = torch.zeros(self.g_dim, self.g_dim).to(device) # 
        self.l_one = torch.zeros(self.g_dim, self.g_dim).to(device) # 
        self.l_two = torch.zeros(self.g_dim, self.g_dim).to(device) # 

        # Storage
        self.error_storage = []
    
    def get_d_hat(self):
        """Get the correlation from the last step"""
        return torch.matmul(self.d_hat, self.u_affine_pinv.transpose(0, 1)) # [N, g_dim] * [u_dim, g_dim]^T -> [N, u_dim]
    
    def set_z1_hat(self, z1_hat_raw):
        """Set the value of z1_hat"""
        assert z1_hat_raw.shape[0] == self.n_envs, f"Expected shape ({self.n_envs}, ...), but got {z1_hat_raw.shape}"
        self.z1_hat = self.lift_transform(z1_hat_raw)
        return
    
    def predict_eso(self, z1_true_raw, u):
        """
        Perform one step of ESO update.

        Args:
            z1_true_raw (Tensor): Current true observation z1(k), shape [N, g_dim]
            u (Tensor): Current control input u(k), shape [N, u_dim]
        Returns:
            N/A
        """
        ### lift raw z1_true
        z1_true = self.lift_transform(z1_true_raw) # [N,g_dim]
        ### specific discrete system
        e1 = z1_true - self.z1_hat # [N,g_dim]
        self.error_storage.append(e1) # storage
        print("error : ", e1[0][0]) # visualize
        ### self-update
        # \dot{\hat{x}} = a \hat{x} + b u + \hat{d} + l_1 (y - \hat{x})
        self.z1_hat = self.predict_kpm(self.z1_hat, u) + self.d_hat + torch.matmul(e1, self.l_one.transpose(0, 1)) # [N, g_dim] -> easy to out of range -> nan
        # \dot{\hat{d}} = l_2 (y - \hat{x})
        self.d_hat = torch.matmul(e1, self.l_two.transpose(0, 1)) # [N, g_dim]
        return True, True, True
    
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



############################################################################################################
# import numpy as np
# import matplotlib.pyplot as plt
# # 系统参数
# a = 0.9        # 状态反馈因子
# b = 0.5        # 控制输入增益
# T = 1.0        # 时间步长
# steps = 500     # 仿真步数
# # 观测器增益（可调）
# L = np.array([[1.8],    # 对应 x
#               [0.2]])   # 对应 d
# # 系统状态、输入、扰动
# x = 0.0
# d = 0.6  # 外部恒定扰动
# u = 1.0
# # ESO 初始值
# z_hat = np.zeros((2,))  # [x_hat, d_hat]
# # 记录
# true_x, est_x, est_d = [], [], []
# for k in range(steps):
#     # 系统演化
#     x = a * x + b * u + d
#     y = x  # 真实输出
#     # ESO 系统矩阵
#     A_z = np.array([[a, 1],
#                     [0, 1]])
#     B_z = np.array([b, 0])
#     # ESO 更新
#     y_hat = z_hat[0]
#     z_hat = A_z @ z_hat + B_z * u + (L.flatten()) * (y - y_hat)
#     # 记录
#     true_x.append(x)
#     est_x.append(z_hat[0])
#     est_d.append(z_hat[1])
# # 画图
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.plot(true_x, label='True x')
# plt.plot(est_x, '--', label='Estimated x')
# plt.title('State Estimation')
# plt.xlabel('Time step'); plt.legend()
# plt.subplot(1, 2, 2)
# plt.plot([d]*steps, label='True d')
# plt.plot(est_d, '--', label='Estimated d')
# plt.title('Disturbance Estimation')
# plt.xlabel('Time step'); plt.legend()
# plt.tight_layout()
# plt.show()
############################################################################################################




