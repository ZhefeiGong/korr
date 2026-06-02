"""
@author: 
| copyright @ Zhefei(Jeffrey) Gong
@date: 
| Feb.25th 2025 -> Mar.19th 2025
@func: 
| the core implementation of Koopman & Linear Quadratic Resolver
@link:
| TO-KPM: https://github.com/xubo92/to-kpm
| Embedding LQR: https://github.com/navigator8972/koopman_policy
"""

import torch
import torch.nn as nn

class KoopmanLQR(nn.Module):
    def __init__(self, 
                 T, 
                 g_dim, 
                 u_dim, 
                 g_goal=None, 
                 g_affine=None,
                 u_affine=None):
        """
        T:          length of horizon
        g_dim:      dimension of latent state
        u_dim:      dimension of control input
        g_goal:     None by default. If not, override the x_goal so it is not necessarily corresponding to a concrete goal state
                    might be useful for non regularization tasks.  
        u_affine:   should be a linear transform for an augmented observation phi(x, u) = phi(x) + nn.Linear(u)
        """
        super().__init__()
        self._T = T # iteration times
        self._g_dim = g_dim # dimension of observation
        self._u_dim = u_dim # dimension of action
        self._g_goal = g_goal # the goal of the LQR learning
        
        # prepare linear system params - g affine matrix
        self._g_affine = g_affine
        if self._g_affine is None:
            self._g_affine = nn.Parameter(torch.empty((self._g_dim, self._g_dim)), requires_grad=True) # [obs_dim, obs_dim] - G
        else:
            self._g_affine = nn.Parameter(self._g_affine, requires_grad=True)
        
        # prepare linear system params - u affine matrix
        self._u_affine = u_affine
        if self._u_affine is None:
            self._u_affine = nn.Parameter(torch.empty((self._g_dim, self._u_dim)), requires_grad=True) # [obs_dim, act_dim] - U
        else:
            self._u_affine = nn.Parameter(self._u_affine, requires_grad=True)
        
        # initialize G and U with Gaussian Distribution
        torch.nn.init.normal_(self._g_affine, mean=0, std=1)
        torch.nn.init.normal_(self._u_affine, mean=0, std=1)
        
        # parameters of quadratic functions -> Symmetric Positive Definite
        self._q_diag_log = nn.Parameter(torch.zeros(self._g_dim), requires_grad=True) # to use: Q = diag(_q_diag_log.exp()) | 
        self._r_diag_log = nn.Parameter(torch.zeros(self._u_dim), requires_grad=True) # to use: Q = diag(_r_diag_log.exp()) | 🤔 it's set to requires_grad=false (in to-kpm) 🤔
        
        # zero tensor constant for k and v in the case of fixed origin
        # these will be automatically moved to gpu so no need to create and check in the forward process
        self.register_buffer('_zero_tensor_constant_k', torch.zeros((1, self._u_dim)))
        self.register_buffer('_zero_tensor_constant_v', torch.zeros((1, self._g_dim)))
        
        # we may need to create a few cache for K, k, V and v because they are not dependent on x
        # unless we make g_goal depend on it. This allows to avoid repeatively calculate riccati recursion in eval mode
        self._riccati_solution_cache = None

        return

    def forward(self, g0):
        """perform mpc with current parameters given the initial x0"""
        K, k, V, v = self._retrieve_riccati_solution() # 
        u = -self._batch_mv(K[0], g0) + k[0] # apply the first control as mpc
        return u
    
    def set_riccati_cache_to_zero(self, device='cpu'):
        """initialize the cache for riccati"""
        K = [torch.zeros(self._u_dim, self._g_dim).to(device) for _ in range(self._T)] # [u_dim, g_dim]
        k = [torch.zeros(self._u_dim).to(device) for _ in range(self._T)] # [u_dim]
        V = [torch.zeros(self._g_dim, self._g_dim).to(device) for _ in range(self._T + 1)] # [g_dim, g_dim]
        v = [torch.zeros(self._g_dim).to(device) for _ in range(self._T + 1)] # [g_dim, g_dim]
        self._riccati_solution_cache = (K,k,V,v)
        return
    
    def set_goal(self,goal=None):
        """set the goal"""
        if goal is not None:
            assert len(goal.shape) == 1, "the dimension of goals is overflow" 
            assert goal.shape[-1] == self._g_dim, "the mismatch dimension of goals"
            self._g_goal = goal
        else:
            self._g_goal = None

    @staticmethod
    def _batch_mv(bmat, bvec):
        """
        Performs a batched matrix-vector product, with compatible but different batch shapes.

        This function takes as input `bmat`, containing :math:`n * n` matrices, and
        `bvec`, containing length :math:`n` vectors.

        Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
        to a batch shape. They are not necessarily assumed to have the same batch shape,
        just ones which can be broadcasted.
        """
        return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)

    def _retrieve_riccati_solution(self):
        """retrieve riccati equation"""

        if self.training or self._riccati_solution_cache is None:
            # load Q,R params
            Q = torch.diag(self._q_diag_log.exp()) # load Q matrix for lqr solver | [obs_dim, obs_dim]
            R = torch.diag(self._r_diag_log.exp()) # load R matrix for lqr solver | [act_dim, act_dim]
            # use g_goal
            if self._g_goal is not None:
                goals = torch.repeat_interleave(self._g_goal.unsqueeze(0), repeats=self._T + 1, dim=0) # [T+1, obs_dim]
            else:
                goals = None
            # solve the lqr problem via a differentiable process.
            K, k, V, v = self._solve_lqr(self._g_affine, self._u_affine, Q, R, goals)
            # store the calculated results
            self._riccati_solution_cache = ([tmp.detach().clone() for tmp in K], 
                                            [tmp.detach().clone() for tmp in k], 
                                            [tmp.detach().clone() for tmp in V], 
                                            [tmp.detach().clone() for tmp in v])
        else:
            K, k, V, v = self._riccati_solution_cache

        return K, k, V, v

    def _solve_lqr(self, A, B, Q, R, goals):
        """linear-quadratic regulator"""

        # @intro:
        # a differentiable process of solving LQR, 
        # time-invariant A, B, Q, R (with leading batch dimensions), but goals can be a batch of trajectories (batch_size, T+1, k)
        #       min \Sigma^{T} (x_t - goal[t])^T Q (x_t - goal[t]) + u_t^T R u_t
        # @return:
        # s.t.  x_{t+1} = A x_t + B u_t
        # return feedback gain and feedforward terms such that u = -K x + k
        # @formula:
        # V_{t} = A^T*V_{t+1}*A − A^T*V_{t+1}*B * (R+B^T*V_{t+1}*B)^{−1}*B^T*V_{t+1}*A + Q
        #   -> V_{t} = A^T*V_{t+1}*A − A^T*V_{t+1}*B * K_{t} + Q
        #   -> V_{t} = A^T*V_{t+1} * (A−B*K_{t}) + Q
        # K_t = (B^T*V_{t+1}*B+R)^{−1}*B^T*V_{t+1}*A

        T = self._T
        K = [None] * T
        k = [None] * T
        V = [None] * (T + 1)
        v = [None] * (T + 1)

        A_trans = A.transpose(-2,-1)
        B_trans = B.transpose(-2,-1)

        V[-1] = Q  # initialization for backpropagation | [obs_dim, obs_dim]
        if goals is not None:
            # Having Goals means a desired point
            v[-1] = self._batch_mv(Q, goals[-1, :]) # [obs_dim,]
            for i in reversed(range(T)):
                # using torch.solve(B, A) to obtain the solution of AX = B to avoid direct inverse, note it also returns LU
                # for new torch.linalg.solve, no LU is returned
                V_uu_inv_B_trans = torch.linalg.solve(torch.matmul(torch.matmul(B_trans, V[i+1]), B) + R, B_trans) # (B^T*V_{t+1}*B+R)^{−1}*B^T -> [act_dim, obs_dim]
                K[i] = torch.matmul(torch.matmul(V_uu_inv_B_trans, V[i+1]), A) # V_uu_inv_B_trans*V_{t+1}*A -> [act_dim, obs_dim]
                k[i] = self._batch_mv(V_uu_inv_B_trans, v[i+1]) # -> [act_dim, ]

                # riccati difference equation, A-BK
                A_BK = A - torch.matmul(B, K[i]) # A−B*K_{t} + Q -> [obs_dim, obs_dim]
                V[i] = torch.matmul(torch.matmul(A_trans, V[i+1]), A_BK) + Q # A^T*V_{t+1} * A_BK + Q -> [obs_dim, obs_dim]
                v[i] = self._batch_mv(A_BK.transpose(-2, -1), v[i+1]) + self._batch_mv(Q, goals[i, :]) # [obs_dim, ]
        else:
            # None goals means a fixed regulation point at origin. ignore k and v for efficiency
            for i in reversed(range(T)):
                # using torch.solve(B, A) to obtain the solution of AX = B to avoid direct inverse, note it also returns LU
                # for new torch.linalg.solve, no LU is returned
                V_uu_inv_B_trans = torch.linalg.solve(torch.matmul(torch.matmul(B_trans, V[i+1]), B) + R, B_trans) # -> [act_dim, obs_dim]
                K[i] = torch.matmul(torch.matmul(V_uu_inv_B_trans, V[i+1]), A) # -> [act_dim, obs_dim]
                
                #riccati difference equation: A-BK
                A_BK = A - torch.matmul(B, K[i]) # -> [obs_dim, obs_dim]
                V[i] = torch.matmul(torch.matmul(A_trans, V[i+1]), A_BK) + Q # -> [obs_dim, obs_dim]
            k[:] = self._zero_tensor_constant_k
            v[:] = self._zero_tensor_constant_v

        # we might need to cat or 
        #  to return them as tensors but for mpc maybe only the first time step is useful...
        # note K is for negative feedback, namely u = -Kx+k
        return K, k, V, v

    def _predict_koopman(self, G, U):
        '''
        predict dynamics with current koopman parameters
        note both input and return are embeddings of the predicted state, we can recover that by using invertible net, e.g. normalizing-flow models
        but that would require a same dimensionality
        '''
        return torch.matmul(G, self._g_affine.transpose(0, 1)) + torch.matmul(U, self._u_affine.transpose(0, 1))



