import numpy as np
import torch
from PINN_Model import LocalVolatilityPINN, sigma_torch
from MLP_Model  import Baseline_MLP

#RMSE: prediction accuracy 
def rmse(a, b): 
    return torch.sqrt(torch.mean((a - b) ** 2))

#MAE: boundary/terminal condition accuracy
def mae(a, b):
    return torch.mean(torch.abs(a - b))

def flatten(y):
    return y.view(-1)

def Normalized_PDE_residual(model, X_hat, K, r, S_min, S_max, t_min, t_max, phi_mean, phi_std):
    X_hat = X_hat.clone().detach().requires_grad_(True)

    S_hat  = X_hat[:, 0:1]
    t_hat  = X_hat[:, 1:2]
    Phi_hat = X_hat[:, 2:7]

    V = model(X_hat)
    if V.ndim == 1:
        V = V.view(-1, 1)

    grad = torch.autograd.grad(
        V, X_hat,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True
    )[0]
    V_S_hat = grad[:, 0:1]
    V_t_hat = grad[:, 1:2]

    grad2 = torch.autograd.grad(
        V_S_hat, X_hat,
        grad_outputs=torch.ones_like(V_S_hat),
        create_graph=True,
        retain_graph=True
    )[0]
    V_ShatShat = grad2[:, 0:1]

    dS_hat_dS = 2.0 / (S_max - S_min + 1e-12)
    dt_hat_dt = 2.0 / (t_max - t_min + 1e-12)

    V_S  = V_S_hat * dS_hat_dS
    V_t  = V_t_hat * dt_hat_dt
    V_SS = V_ShatShat * (dS_hat_dS ** 2)

    S = (S_hat + 1.0) * 0.5 * (S_max - S_min) + S_min
    t = (t_hat + 1.0) * 0.5 * (t_max - t_min) + t_min

    Phi = Phi_hat * phi_std + phi_mean

    sig = sigma_torch(S, t, Phi, K)
    if sig.ndim == 1:
        sig = sig.view(-1, 1)

    residual = V_t + 0.5 * (sig ** 2) * (S ** 2) * V_SS + r * S * V_S - r * V
    return residual


def pde_rms_batched(model, X_hat, K, r, S_min, S_max, t_min, t_max, phi_mean, phi_std,
                    batch_size=256):
    """
    RMS(PDE residual) computed in batches to avoid GPU OOM.
    """
    model.eval()
    n = X_hat.shape[0]
    sum_sq = torch.zeros((), device=X_hat.device)
    count = 0

    for i in range(0, n, batch_size):
        xb = X_hat[i:i + batch_size]
        res = Normalized_PDE_residual(model, xb, K, r, S_min, S_max, t_min, t_max, phi_mean, phi_std)
        sum_sq += torch.sum(res ** 2).detach()
        count += res.numel()

        # free graph asap
        del res
        if X_hat.device.type == "cuda":
            torch.cuda.empty_cache()

    return torch.sqrt(sum_sq / count)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    d = np.load("Localvol_dataset_test_normalized.npz", allow_pickle=True)

    X_int = d["X_int_test"].astype(np.float32)
    y_int = d["y_int_test"].astype(np.float32).reshape(-1)

    X_bc  = d["X_bc_test"].astype(np.float32)
    y_bc  = d["y_bc_test"].astype(np.float32).reshape(-1)

    X_tm  = d["X_term_test"].astype(np.float32)
    y_tm  = d["y_term_test"].astype(np.float32).reshape(-1)

    X_int_t = torch.from_numpy(X_int).to(device)
    X_bc_t  = torch.from_numpy(X_bc).to(device)
    X_tm_t  = torch.from_numpy(X_tm).to(device)

    y_int_t = torch.from_numpy(y_int).to(device)
    y_bc_t  = torch.from_numpy(y_bc).to(device)
    y_tm_t  = torch.from_numpy(y_tm).to(device)

    phi_mean_t = torch.tensor(d["phi_mean"].astype(np.float32), device=device).view(1, 5)
    phi_std_t  = torch.tensor(d["phi_std"].astype(np.float32),  device=device).view(1, 5)

    S_min_t = torch.tensor(float(np.array(d["S_min"]).reshape(-1)[0]), device=device)
    S_max_t = torch.tensor(float(np.array(d["S_max"]).reshape(-1)[0]), device=device)
    t_min_t = torch.tensor(float(np.array(d["t_min"]).reshape(-1)[0]), device=device)
    t_max_t = torch.tensor(float(np.array(d["t_max"]).reshape(-1)[0]), device=device)
    K_t     = torch.tensor(float(np.array(d["K"]).reshape(-1)[0]),     device=device)
    r_t     = torch.tensor(float(np.array(d["r"]).reshape(-1)[0]),     device=device)

    pinn = LocalVolatilityPINN(width=64, depth=8, activation="tanh").to(device)
    ckpt_pinn = torch.load("PINN_model.pt", map_location=device)
    pinn.load_state_dict(ckpt_pinn["model_state"])
    pinn.eval()

    mlp = Baseline_MLP(width=64, depth=8, activation="tanh").to(device)
    ckpt_mlp = torch.load("MLP_model.pt", map_location=device)
    mlp.load_state_dict(ckpt_mlp["model_state"])
    mlp.eval()

    with torch.no_grad():
        yhat_int_pinn = flatten(pinn(X_int_t))
        yhat_int_mlp  = flatten(mlp(X_int_t))

        yhat_bc_pinn  = flatten(pinn(X_bc_t))
        yhat_bc_mlp   = flatten(mlp(X_bc_t))

        yhat_tm_pinn  = flatten(pinn(X_tm_t))
        yhat_tm_mlp   = flatten(mlp(X_tm_t))

    int_rmse_pinn = rmse(yhat_int_pinn, y_int_t)
    int_rmse_mlp  = rmse(yhat_int_mlp,  y_int_t)

    bc_mae_pinn   = mae(yhat_bc_pinn, y_bc_t)
    bc_mae_mlp    = mae(yhat_bc_mlp,  y_bc_t)

    term_mae_pinn = mae(yhat_tm_pinn, y_tm_t)
    term_mae_mlp  = mae(yhat_tm_mlp,  y_tm_t)

    max_pde_points = 5000
    pde_batch_size = 128

    if X_int_t.shape[0] > max_pde_points:
        idx = torch.randperm(X_int_t.shape[0], device=device)[:max_pde_points]
        X_pde = X_int_t[idx]
    else:
        X_pde = X_int_t

    pde_rms_pinn = pde_rms_batched(
        pinn, X_pde, K_t, r_t, S_min_t, S_max_t, t_min_t, t_max_t, phi_mean_t, phi_std_t,
        batch_size=pde_batch_size
    )
    pde_rms_mlp = pde_rms_batched(
        mlp, X_pde, K_t, r_t, S_min_t, S_max_t, t_min_t, t_max_t, phi_mean_t, phi_std_t,
        batch_size=pde_batch_size
    )
    
    delta_V = torch.max(y_int_t) - torch.min(y_int_t)
    scale = delta_V * 0.5

    RMSE_orig_pinn = int_rmse_pinn * scale
    RMSE_orig_mlp  = int_rmse_mlp  * scale
    MAE_orig_boundary_pinn = bc_mae_pinn * scale
    MAE_orig_boundary_mlp  = bc_mae_mlp  * scale
    MAE_orig_terminal_pinn = term_mae_pinn * scale
    MAE_orig_terminal_mlp  = term_mae_mlp  * scale

print("\nEvaluation Metrics:")

label_w = 18
model_w = 18

print(f"{'Interior RMSE':<{label_w}} | {'PINN: ' + format(int_rmse_pinn.item(), '.6f'):<{model_w}} | {'MLP: ' + format(int_rmse_mlp.item(), '.6f')}")
print(f"{'Boundary MAE':<{label_w}} | {'PINN: ' + format(bc_mae_pinn.item(), '.6f'):<{model_w}} | {'MLP: ' + format(bc_mae_mlp.item(), '.6f')}")
print(f"{'Terminal MAE':<{label_w}} | {'PINN: ' + format(term_mae_pinn.item(), '.6f'):<{model_w}} | {'MLP: ' + format(term_mae_mlp.item(), '.6f')}")
print(f"{'PDE residual RMS*':<{label_w}} | {'PINN: ' + format(pde_rms_pinn.item(), '.6f'):<{model_w}} | {'MLP: ' + format(pde_rms_mlp.item(), '.6f')}")
print(f"*PDE RMS computed on {X_pde.shape[0]} interior points (batch_size={pde_batch_size}).")
print("")
print(f"Original scale (un-normalized) metrics:")
print(f"{'Interior RMSE':<{label_w}} | {'PINN: ' + format(RMSE_orig_pinn.item(), '.6f'):<{model_w}} | {'MLP: ' + format(RMSE_orig_mlp.item(), '.6f')}")
print(f"{'Boundary MAE':<{label_w}} | {'PINN: ' + format(MAE_orig_boundary_pinn.item(), '.6f'):<{model_w}} | {'MLP: ' + format(MAE_orig_boundary_mlp.item(), '.6f')}")
print(f"{'Terminal MAE':<{label_w}} | {'PINN: ' + format(MAE_orig_terminal_pinn.item(), '.6f'):<{model_w}} | {'MLP: ' + format(MAE_orig_terminal_mlp.item(), '.6f')}")

