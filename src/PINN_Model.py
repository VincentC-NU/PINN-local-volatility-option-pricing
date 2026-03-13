import torch
import torch.nn as nn

class LocalVolatilityPINN(nn.Module):
    def __init__(self, input_dim=7, width=64, depth=8, activation="tanh"):
        super().__init__()
        if isinstance(activation, str):
            act = nn.Tanh if activation.lower() == "tanh" else nn.ReLU
        else:
            act = activation

        layers = [nn.Linear(input_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

#Differntiable reconstruction of local volatility function for PINN training
def sigma_torch(S, t, Phi, K): 
    eps = 1e-8
    S_safe = torch.clamp(S, min=eps)

    sigma0 = Phi[:, 0:1]
    alpha  = Phi[:, 1:2]
    beta   = Phi[:, 2:3]
    gamma  = Phi[:, 3:4]
    eta    = Phi[:, 4:5]
    
    log_m = torch.log(S_safe / K)
    skew = 1.0 + alpha * torch.tanh(beta * log_m)
    term = 1.0 + gamma * torch.exp(-eta * t)
    return sigma0 * skew * term

import torch

def unnormalize_st(x_hat, xmin, xmax):
    return xmin + 0.5 * (x_hat + 1.0) * (xmax - xmin)

def Normalized_PDE_residual(model, X_hat, K, r, S_min, S_max, t_min, t_max, phi_mean, phi_std):
    """
    X_hat: normalized inputs (B,7) = [S_hat, t_hat, Phi_hat(5)]
    Returns residual of "physical" local-vol BS PDE.
    """
    X_hat = X_hat.clone().requires_grad_(True)

    S_hat = X_hat[:, 0:1]
    t_hat = X_hat[:, 1:2]
    Phi_hat = X_hat[:, 2:7]
    
    V = model(X_hat)
    
    #first-order gradient with normalzied input
    grad = torch.autograd.grad(
        V, X_hat, grad_outputs=torch.ones_like(V),
        create_graph=True, retain_graph=True
    )[0]
    V_S_hat = grad[:, 0:1]
    V_t_hat = grad[:, 1:2]

    #second-order gradient with normalized input
    grad2 = torch.autograd.grad(
        V_S_hat, X_hat, grad_outputs=torch.ones_like(V_S_hat),
        create_graph=True, retain_graph=True
    )[0]
    V_ShatShat = grad2[:, 0:1]

    #convert derivatives to physical using chain rule
    dS_hat_dS = 2.0 / (S_max - S_min)
    dt_hat_dt = 2.0 / (t_max - t_min)

    V_S  = V_S_hat * dS_hat_dS
    V_t  = V_t_hat * dt_hat_dt
    V_SS = V_ShatShat * (dS_hat_dS ** 2)

    #rconstruct physical S,t
    S = unnormalize_st(S_hat, S_min, S_max)
    t = unnormalize_st(t_hat, t_min, t_max)

    #unstandardize Phi for sigma computation
    #phi_mean, phi_std should be torch tensors on the right device
    Phi = Phi_hat * phi_std + phi_mean

    sig = sigma_torch(S, t, Phi, K)

    residual = V_t + 0.5 * (sig**2) * (S**2) * V_SS + r * S * V_S - r * V
    return residual