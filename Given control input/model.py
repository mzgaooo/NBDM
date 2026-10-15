import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class NBDM(nn.Module):
    def __init__(self, input_dim, h_hidden_dim, u_hidden_dim, num_nodes, pred_len, dropout, A_edmd, device):
        super().__init__()
        self.input_dim = input_dim
        self.h_hidden_dim = h_hidden_dim
        self.u_hidden_dim = u_hidden_dim
        self.num_nodes = num_nodes
        self.pred_steps = pred_len
        self.dropout = dropout
        self.device = device
        self.linear_1 = nn.Linear(4*h_hidden_dim, h_hidden_dim)
        self.x_generate = nn.Linear(h_hidden_dim + u_hidden_dim, h_hidden_dim + h_hidden_dim*u_hidden_dim + 
                                    h_hidden_dim*h_hidden_dim*u_hidden_dim + h_hidden_dim)
        
        # ========== Predifined EDMD Graph ==========
        self.A_edmd = A_edmd
        self.A_edmd_norm = nn.BatchNorm1d(num_nodes)

        # ========== Embedding Layer ==========
        self.x_project = nn.Sequential(
            nn.Linear(input_dim, h_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(h_hidden_dim // 2, h_hidden_dim)
        )

        self.u_project = nn.Sequential(
            nn.Linear(input_dim, u_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(u_hidden_dim // 2, u_hidden_dim)
        )

        # ========== Error Model Parameters  ==========
        self.error_term = nn.Parameter(
            torch.randn(num_nodes, h_hidden_dim) * 0.01,
            requires_grad=True
        )
        
        # ========== Normalization Layers ==========
        self.hidden_norm = nn.LayerNorm(h_hidden_dim)
        
        # ========== Initialization ==========
        self._init_parameters()
    

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)



    def bilinear_message_passing(self, h, u, A, B0, B_tilde):
        B, N, hd = h.shape
        
        # ========== Linear state transition ==========
        # A * h term
        h_linear = A * h  # (B, N, hd)
        
        # ========== Control input term ==========
        # B0 * u term
        B0 = B0.reshape(B,N,hd,-1)
        u_control = torch.einsum('bndm,bnm->bnd', B0, u)  # (B, N, hd)
        
        # ========== Bilinear term (if enabled) ==========
        u_expanded = u.unsqueeze(-1)  # (B, N, ud, 1)
        h_expanded = h.unsqueeze(-2)  # (B, N, 1, hd)
        u_kronecker_h = (u_expanded * h_expanded).reshape(B, N, -1)   #(B,N,ud*hd)

        B_tilde = B_tilde.reshape(B,N,hd,-1)  #(B,N,hd,hd*ud)
        bilinear_term = torch.einsum('bnij,bnj->bni', B_tilde, u_kronecker_h)  # (B, N, hd)

        error = self.error_term.expand(B, N, hd)
        
        # ========== Combine all terms ==========
        h_next = torch.cat([h_linear, u_control, bilinear_term, error], dim=-1)
        h_next = self.linear_1(h_next)
        h_next = self.hidden_norm(h_next)
        
        return h_next, error


    def forward(self, x, u):
        x = x.permute(0,2,1) # (B, N, T)
        h_current = self.x_project(x)  # (B, N, hd)
        B, N, hd = h_current.shape

        u = u.permute(0,2,1)
        ebed_u = self.u_project(u)  # (B, N, ud)
        B, N, ud = ebed_u.shape

        h_predictions = []
        error_violations = []

        for step in range(self.pred_steps):
            hid_data = torch.cat((h_current,ebed_u),dim=-1)
            hid_data = self.x_generate(hid_data) #  B N hd+ud -> B N hd+hd*ud+hd*hd*ud+hd
            (A_delta, B0, B_tilde, C) = hid_data.split(split_size=[hd, hd*ud, hd*hd*ud, hd], dim=-1)

            # Bilinear message passing
            A_edmd_norm = self.A_edmd_norm(self.A_edmd)
            A = torch.einsum('bnd,nm->bmd', A_delta, A_edmd_norm)
            h_next, error = self.bilinear_message_passing(h_current, ebed_u, A, B0, B_tilde)
            h_current = h_next

            C = C.reshape(B,N,-1,self.h_hidden_dim)
            h_prediction = torch.einsum('bnmd,bnd->bnm', C, h_next)   #BN1D   BND  BN1
            h_predictions.append(h_prediction.unsqueeze(1))
            
        # Project back to original space
        predictions = torch.cat(h_predictions, dim=1)  # (B, T_pred, N, 1)

        return predictions