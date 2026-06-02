"""
@author: 
| copyright @ Zhefei(Jeffrey) Gong
@date: 
| Mar.30th 2025
@func: 
| the core implementation of Linear Quadratic Regulator
@link:
| TO-KPM: https://github.com/xubo92/to-kpm
| Embedding LQR: https://github.com/navigator8972/koopman_policy
"""

import copy
import torch
import torch.nn as nn

class LinearQuadraticRegulator:
    def __init__(self, 
                 T, 
                 g_affine=None,
                 u_affine=None,
                 lift_transform=None,
                 device='cpu'):
        """
        T:          length of horizon
        g_dim:      dimension of latent state
        u_dim:      dimension of control input
        g_goal:     None by default. If not, override the x_goal so it is not necessarily corresponding to a concrete goal state
                    might be useful for non regularization tasks.  
        g_affine:   should be a linear transform for an augmented observation phi(x, u) = phi(x) + nn.Linear(u)
        u_affine:   should be a linear transform for an augmented observation phi(x, u) = phi(x) + nn.Linear(u)
        """
        super().__init__()

        # prepare linear system params - affine matrix
        assert (g_affine is not None) and (u_affine is not None), "receive the g_affine or u_affine with value of None"

        # g_affine & normalization parameters
        self._g_affine = torch.Tensor(g_affine).to(device) # [g_dim, g_dim]
        self._g_mean = self._g_affine.mean()
        self._g_std = self._g_affine.std()
        self._g_max = self._g_affine.max()
        self._g_min = self._g_affine.min()

        # u_affine & normalization parameters
        self._u_affine = torch.Tensor(u_affine).to(device) # [g_dim, u_dim]
        self._u_mean = self._u_affine.mean()
        self._u_std = self._u_affine.std()
        self._u_max = self._u_affine.max()
        self._u_min = self._u_affine.min()

        # initialize other params
        self._T = T # iteration times
        self._g_dim = self._g_affine.shape[0] # dimension of observation
        self._u_dim = self._u_affine.shape[-1] # dimension of action
        self._device = device

        ##### data record from XinLang
        # q - [200*7, 2*7]
        # r - [0.1*7]
        ##### data tries here
        # e^5 = 148.4131591 | 
        # e^(-2) = 0.13533528 | 

        # parameters of quadratic functions -> Symmetric Positive Definite
        self._q_diag_log = torch.full((self._g_dim,), -5.0).to(device) # to use: Q = diag(_q_diag_log.exp())
        self._r_diag_log = torch.full((self._u_dim,), 10.0).to(device) # to use: Q = diag(_r_diag_log.exp())

        # up-transform the dimension
        self.lift_transform = lift_transform

        return
    
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
    
    def _retrieve_riccati_solution(self, goal):
        """retrieve riccati equation"""
        # load Q,R params
        Q = torch.diag(self._q_diag_log.exp()) # load Q matrix for lqr solver | [obs_dim, obs_dim] | e^x
        R = torch.diag(self._r_diag_log.exp()) # load R matrix for lqr solver | [act_dim, act_dim] | e^x
        # solve the lqr problem via a differentiable process.
        K, k, V, v = self._solve_lqr(self._g_affine, self._u_affine, Q, R, goal)
        return K, k, V, v
    
    def _normalize_minmax(self, tensor, v_min=-1.0, v_max=1.0):
        """Normalize tensor using min and max"""
        normalized_tensor = (tensor - v_min) / (v_max - v_min)  # -> [0, 1]
        normalized_tensor = 2 * normalized_tensor - 1  # -> [-1, 1]
        return normalized_tensor
    
    def _denormalize_minmax(self, tensor, v_min=-1.0, v_max=1.0):
        """Denormalize tensor using min and max"""
        denormalized_tensor = (tensor + 1) / 2 # -> [0, 1]
        denormalized_tensor = denormalized_tensor * (v_max - v_min) + v_min # -> [v_min, v_max]
        return denormalized_tensor
    
    def _solve_lqr(self, A, B, Q, R, goal=None):
        """
        @name:
            linear-quadratic regulator
        @intro:
            a differentiable process of solving LQR, 
            time-invariant A, B, Q, R (with leading batch dimensions), but goals can be a batch of trajectories (batch_size, T+1, k)
              min \Sigma^{T} (x_t - goal[t])^T Q (x_t - goal[t]) + u_t^T R u_t
        @return:
            s.t.  x_{t+1} = A x_t + B u_t
            return feedback gain and feedforward terms such that u = -K x + k
        @formula (for traditional LQR):
            V_{t} = A^T*V_{t+1}*A - A^T*V_{t+1}*B * (R+B^T*V_{t+1}*B)^{-1}*B^T*V_{t+1}*A + Q
                -> V_{t} = A^T*V_{t+1}*A - A^T*V_{t+1}*B * K_{t} + Q
                -> V_{t} = A^T*V_{t+1} * (A-B*K_{t}) + Q
            K_t = (B^T*V_{t+1}*B+R)^{-1}*B^T*V_{t+1}*A"
        """

        # initialization for backpropagation | [obs_dim, obs_dim]
        T = self._T
        K = None # -> [u_dim, g_dim]
        k = None # -> [u_dim] for goal
        V = None # -> [g_dim, g_dim]
        v = None # -> [g_dim, g_dim] for goal
        A_trans = A.transpose(-2,-1)
        B_trans = B.transpose(-2,-1)
        V = copy.deepcopy(Q) # deepcopy here

        if goal is not None:
            # Having Goals means a desired point
            v = self._batch_mv(Q, goal) # [obs_dim,]
            for i in reversed(range(T)):
                # using torch.solve(B, A) to obtain the solution of AX = B to avoid direct inverse, note it also returns LU
                # for new torch.linalg.solve, no LU is returned
                V_uu_inv_B_trans = torch.linalg.solve(torch.matmul(torch.matmul(B_trans, V), B) + R, B_trans) # (B^T*V_{t+1}*B+R)^{−1}*B^T -> [act_dim, obs_dim]
                K = torch.matmul(torch.matmul(V_uu_inv_B_trans, V), A) # V_uu_inv_B_trans*V_{t+1}*A -> [act_dim, obs_dim]
                k = self._batch_mv(V_uu_inv_B_trans, v) # V_uu_inv_B_trans*v_{t+1} -> [act_dim, ]

                # riccati difference equation, A-BK
                A_BK = A - torch.matmul(B, K) # A − B * K_{t} + Q -> [obs_dim, obs_dim]
                V_new = torch.matmul(torch.matmul(A_trans, V), A_BK) + Q # A^T * V_{t+1} * A_BK + Q -> [obs_dim, obs_dim]
                print(torch.linalg.norm(V_new - V, ord='fro'))
                V = V_new
                v = self._batch_mv(A_BK.transpose(-2, -1), v) + self._batch_mv(Q, goal) # A_BK^T * v_{t+1} + Q*g_t -> [obs_dim, ]
        else:
            # None goals means a fixed regulation point at origin. ignore k and v for efficiency
            for i in reversed(range(T)):
                # using torch.solve(B, A) to obtain the solution of AX = B to avoid direct inverse, note it also returns LU
                # for new torch.linalg.solve, no LU is returned
                V_uu_inv_B_trans = torch.linalg.solve(torch.matmul(torch.matmul(B_trans, V), B) + R, B_trans) # -> [act_dim, obs_dim]
                K = torch.matmul(torch.matmul(V_uu_inv_B_trans, V), A) # -> [act_dim, obs_dim]

                # riccati difference equation: A-BK
                A_BK = A - torch.matmul(B, K) # -> [obs_dim, obs_dim]
                V = torch.matmul(torch.matmul(A_trans, V), A_BK) + Q # -> [obs_dim, obs_dim]

            k = torch.zeros(self._u_dim).to(self._device) # [u_dim]
            v = torch.zeros(self._g_dim).to(self._device) # [g_dim]

        # we might need to cat or 
        #  to return them as tensors but for mpc maybe only the first time step is useful...
        # note K is for negative feedback, namely u = -Kx+k
        return K, k, V, v
    
    def compute_controllability_matrix(self, A, B):
        """
        Compute the controllability matrix C = [B, AB, A^2B, ..., A^(n-1)B]
        and check its rank.
        """
        n = A.shape[0]  # State dimension
        C = B  # First column: B
        # Compute [B, AB, A^2B, ..., A^(n-1)B]
        for i in range(1, n):
            C = torch.cat((C, torch.matmul(A, C[:, -B.shape[1]:])), dim=1)  # Append A^i B
        # Compute rank
        rank_C = torch.linalg.matrix_rank(C)
        # Check if system is controllable
        if rank_C == n:
            print("The system is controllable ✅")
        else:
            print("The system is NOT controllable ❌")
        return C, rank_C
    
    def compute_controllability_gramian(self, A, B):
        """
        Compute the controllability Gramian matrix W_c = sum(A^k B B^T (A^k)^T)
        and check its rank.
        """
        n = A.shape[0]  # state dimension
        W_c = torch.zeros((n, n)).to(self._device)  # initialize Gramian matrix
        # compute W_c = sum(A^k B B^T (A^k)^T)
        Ak = torch.eye(n).to(self._device)  # A^0 = I
        for _ in range(n):
            W_c += Ak @ B @ B.T @ Ak.T
            Ak = A @ Ak  # compute next A^k
        # compute rank of Wc
        rank_Wc = torch.linalg.matrix_rank(W_c)
        # Check if system is controllable
        if rank_Wc == n:
            print("The system is controllable ✅")
        else:
            print("The system is NOT controllable ❌")
        return W_c, rank_Wc
    
    def compute_matrix(self, A):
        rank = torch.linalg.matrix_rank(A) # -> n
        print('rank :', rank)
        determinant = torch.linalg.det(A) # -> low
        print('det :', determinant)
        eigv = torch.linalg.eigvals(A) # from eigen value (Eigendecomposition) -> <1
        print('eigen values :', eigv)
        cond = torch.linalg.cond(A) # from singular value (SVD) -> 1
        print('cond :', cond)
        return
    
    def predict_kpm(self, G, U):
        '''
        predict dynamics with current koopman parameters
        note both input and return are embeddings of the predicted state, we can recover that by using invertible net, e.g. normalizing-flow models
        but that would require a same dimensionality
        '''
        G = torch.Tensor(G).to(self._device)
        G = self.lift_transform(G)
        U = torch.Tensor(U).to(self._device)
        return torch.matmul(G, self._g_affine.transpose(0, 1)) + torch.matmul(U, self._u_affine.transpose(0, 1))
    
    def predict_lqr(self, obs, goal):
        """perform mpc with current parameters given the initial x0"""
        assert len(obs.shape) == len(goal.shape) == 2, "the dimension of obs or goal is overflow"
        obs = torch.Tensor(obs).to(self._device)
        obs = self.lift_transform(obs)
        goal = torch.Tensor(goal).to(self._device)

        # # self.compute_controllability_gramian(self._g_affine, self._u_affine)
        # A = self._normalize_minmax(self._g_affine, self._g_min, self._g_max)
        # B = self._normalize_minmax(self._u_affine, self._u_min, self._u_max)
        # self.compute_A(A)
        self.compute_matrix(self._g_affine)

        # goal = self.lift_transform(goal) # already lifted through Koopman derivation
        K, k, V, v = self._retrieve_riccati_solution(goal) # K, k
        u = -self._batch_mv(K, obs) + k # u = -K x + k
        
        return u
    



