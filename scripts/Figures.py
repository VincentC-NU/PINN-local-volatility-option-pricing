import numpy as np
import matplotlib.pyplot as plt
import torch
from Local_volatility_function import local_volatility
from Crank_Nicolson_PDE_solver import crank_nicolson_solver
from Data_normalization import Normalization
from MLP_Model import Baseline_MLP
from PINN_Model import LocalVolatilityPINN, Normalized_PDE_residual

PINN_log = np.load("PINN_training_logs.npz", allow_pickle=True)
MLP_log  = np.load("MLP_training_logs.npz", allow_pickle=True)

PINN_train_loss = PINN_log["train_losses"].astype(float)
PINN_val_loss = PINN_log["val_total"].astype(float)
PINN_pde_loss = PINN_log["pde"].astype(float)

MLP_train_loss = MLP_log["train_losses"].astype(float)
MLP_val_loss = MLP_log["val_total"].astype(float)

#PINN loss curves

plt.figure(figsize=(10,5))
plt.grid(True, alpha=0.3)
plt.semilogy(PINN_train_loss, label="PINN Training Loss", linewidth=2)
plt.semilogy(PINN_val_loss, label="PINN Validation Loss", linewidth=2)
plt.semilogy(PINN_pde_loss, label="PINN PDE Residual Loss", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Log10 Loss")
plt.legend(frameon=False)
plt.grid(True)
plt.savefig("PINN_loss_curves.png", dpi=300, bbox_inches="tight")
plt.show()


#MLP loss curves
plt.figure(figsize=(10,5))
plt.grid(True, alpha=0.3)
plt.semilogy(MLP_train_loss, label="MLP Training Loss", linewidth=2)
plt.semilogy(MLP_val_loss, label="MLP Validation Loss", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Log10 Loss")
plt.legend(frameon=False)
plt.grid(True)
plt.savefig("MLP_loss_curves.png", dpi=300, bbox_inches="tight")
plt.show()

#Sample local volatility and option price heatmaps from the training data
test_data = np.load("Localvol_dataset_test.npz", allow_pickle=True)
S_grid = test_data["S_grid"].astype(float)
t_grid = test_data["t_grid"].astype(float)

sample_Phi = test_data["phis"][0]
sample_sigma = local_volatility(test_data["S_grid"], test_data["t_grid"], sample_Phi)
K = float(test_data["K"])
r = float(test_data["r"])

sample_V = crank_nicolson_solver(test_data["S_grid"], 
                                 test_data["t_grid"], 
                                 sample_sigma, 
                                 K=K, 
                                 r=r, 
                                 option_type=test_data["option_type"])

S_mesh, t_mesh = np.meshgrid(S_grid, t_grid)

plt.figure()

contour = plt.contourf(
    S_mesh,
    t_mesh,
    sample_V.T,  
    levels=40,
    cmap="viridis"
)

plt.contour(
    S_mesh,
    t_mesh,
    sample_V.T,
    levels=20,
    colors="black",
    linewidths=0.3
)

plt.colorbar(contour, label=r"$V(S,t)$")
plt.xlabel(r"$S$")
plt.ylabel(r"$t$")
plt.tight_layout()
plt.savefig("True_option_price_surface.png", dpi=300, bbox_inches="tight")
plt.show()


#Local volatility surface predicted by PINN and MLP
def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        torch.cuda.init()
        _ = torch.empty(1, device="cuda")
        torch.cuda.synchronize()
    return device

device = get_device()

S_flat = S_mesh.ravel()
t_flat = t_mesh.ravel()

Phi_repeat = np.tile(sample_Phi, (S_flat.shape[0], 1))

X_eval = np.column_stack([
    S_flat,
    t_flat,
    Phi_repeat
])

S_min = float(test_data["S_min"])
S_max = float(test_data["S_max"])
t_min = float(test_data["t_min"])
t_max = float(test_data["t_max"])
phi_mean = test_data["phi_mean"]
phi_std = test_data["phi_std"]

S_norm = Normalization(X_eval[:,0], S_min, S_max)
t_norm = Normalization(X_eval[:,1], t_min, t_max)

Phi_norm = (X_eval[:,2:7] - phi_mean) / phi_std

y_all = np.concatenate([
    test_data["y_int"],
    test_data["y_bc"],
    test_data["y_term"]
])
y_min = float(y_all.min())
y_max = float(y_all.max())

X_norm = np.column_stack([S_norm, t_norm, Phi_norm])
X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)


PINN = LocalVolatilityPINN(width=64, depth=8, activation="tanh").to(device)
ckpt_pinn = torch.load("PINN_model.pt", map_location=device)
PINN.load_state_dict(ckpt_pinn["model_state"])
PINN.eval()

MLP = Baseline_MLP(width=64, depth=8, activation="tanh").to(device)
ckpt_mlp = torch.load("MLP_model.pt", map_location=device)
MLP.load_state_dict(ckpt_mlp["model_state"])
MLP.eval()

with torch.no_grad():
    PINN_pred_norm = PINN(X_tensor).cpu().numpy().flatten()
    MLP_pred_norm  = MLP(X_tensor).cpu().numpy().flatten()

K_scale = float(np.array(test_data["K"]).reshape(-1)[0])

PINN_pred = PINN_pred_norm * K_scale
MLP_pred  = MLP_pred_norm  * K_scale

Nt, Ns = len(t_grid), len(S_grid)
V_pinn = PINN_pred.reshape(Nt, Ns)
V_mlp  = MLP_pred.reshape(Nt, Ns)

# Plot contour surfaces
levels = 50

plt.figure(figsize=(7, 5))
cs = plt.contourf(S_mesh, t_mesh, V_pinn, levels=levels)
plt.contour(S_mesh, t_mesh, V_pinn, levels=20, colors="black", linewidths=0.3)
plt.colorbar(cs, label=r"$V_{\mathrm{PINN}}(S,t)$")
plt.xlabel(r"$S$")
plt.ylabel(r"$t$")
plt.tight_layout()
plt.savefig("PINN_prediction.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(7, 5))
cs = plt.contourf(S_mesh, t_mesh, V_mlp, levels=levels)
plt.contour(S_mesh, t_mesh, V_mlp, levels=20, colors="black", linewidths=0.3)
plt.colorbar(cs, label=r"$V_{\mathrm{MLP}}(S,t)$")
plt.xlabel(r"$S$")
plt.ylabel(r"$t$")
plt.tight_layout()
plt.savefig("MLP_prediction.png", dpi=300, bbox_inches="tight")
plt.show()


#Error maps
V_true = sample_V.T
PINN_error = np.abs(V_pinn - V_true)
MLP_error  = np.abs(V_mlp  - V_true)

levels = 40
vmax = max(PINN_error.max(), MLP_error.max())

# PINN
plt.figure(figsize=(7,5))
cs = plt.contourf(S_mesh, t_mesh, PINN_error, levels=levels, vmin=0, vmax=vmax)
plt.contour(S_mesh, t_mesh, PINN_error, levels=15, colors="black", linewidths=0.3)
plt.colorbar(cs, label=r"$|V_{\mathrm{PINN}} - V_{\mathrm{True}}|$")
plt.xlabel("S"); plt.ylabel("t")
plt.tight_layout()
plt.savefig("PINN_error.png", dpi=300, bbox_inches="tight")
plt.show()

# MLP
plt.figure(figsize=(7,5))
cs = plt.contourf(S_mesh, t_mesh, MLP_error, levels=levels, vmin=0, vmax=vmax)
plt.contour(S_mesh, t_mesh, MLP_error, levels=15, colors="black", linewidths=0.3)
plt.colorbar(cs, label=r"$|V_{\mathrm{MLP}} - V_{\mathrm{True}}|$")
plt.xlabel("S"); plt.ylabel("t")
plt.tight_layout()
plt.savefig("MLP_error.png", dpi=300, bbox_inches="tight")
plt.show()

#PDE residual plot
phi_mean_t = torch.tensor(phi_mean, dtype=torch.float32, device=device).view(1, 5)
phi_std_t  = torch.tensor(phi_std,  dtype=torch.float32, device=device).view(1, 5)

R = Normalized_PDE_residual(
    PINN, X_tensor,
    K=K, r=r,
    S_min=S_min, S_max=S_max,
    t_min=t_min, t_max=t_max,
    phi_mean=phi_mean_t,
    phi_std=phi_std_t
)

R = R.detach().cpu().numpy().reshape(Nt, Ns)

R_abs = np.abs(R)
R_log = np.log10(R_abs + 1e-12)

levels = 40

fig, ax = plt.subplots(figsize=(7, 5))

cf = ax.contourf(S_mesh, t_mesh, R_log, levels=levels)
ax.contour(S_mesh, t_mesh, R_log, levels=15, colors="black", linewidths=0.3)

cbar = fig.colorbar(cf, ax=ax)
cbar.set_label(r"$\log_{10}(|\mathcal{R}(S,t)|)$")

ax.set_xlabel(r"$S$")
ax.set_ylabel(r"$t$")

fig.tight_layout()
fig.savefig("PINN_PDE_residual_log.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)
