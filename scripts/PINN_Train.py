import numpy as np
import torch
import torch.nn as nn
import time
from Data_normalization import make_loader
from PINN_Model import LocalVolatilityPINN, Normalized_PDE_residual

data = np.load("Localvol_dataset_train_normalized_noisy.npz", allow_pickle=True)

X_int_tr = data["X_int_train"]
y_int_tr = data["y_int_train"]
X_int_va = data["X_int_val"]
y_int_va = data["y_int_val"]

X_bc_tr  = data["X_bc_train"]
y_bc_tr  = data["y_bc_train"]
X_bc_va  = data["X_bc_val"]
y_bc_va  = data["y_bc_val"]

X_tm_tr  = data["X_term_train"]
y_tm_tr  = data["y_term_train"]
X_tm_va  = data["X_term_val"]
y_tm_va  = data["y_term_val"]

K = float(data["K"][0])
r = float(data["r"][0])

S_min = float(data["S_min"][0])
S_max = float(data["S_max"][0])
t_min = float(data["t_min"][0])
t_max = float(data["t_max"][0])

phi_mean = data["phi_mean"]
phi_std  = data["phi_std"]

stats = {
    "S_min": S_min,
    "S_max": S_max,
    "t_min": t_min,
    "t_max": t_max,
    "phi_mean": phi_mean,
    "phi_std": phi_std,
}

batch_size = 4096

ld_int = make_loader(X_int_tr, y_int_tr, batch_size, shuffle=True,  drop_last=True)
ld_bc  = make_loader(X_bc_tr,  y_bc_tr,  batch_size, shuffle=True,  drop_last=True)
ld_tm  = make_loader(X_tm_tr,  y_tm_tr,  batch_size, shuffle=True,  drop_last=True)

ld_int_val = make_loader(X_int_va, y_int_va, batch_size, shuffle=False, drop_last=False)
ld_bc_val  = make_loader(X_bc_va,  y_bc_va,  batch_size, shuffle=False, drop_last=False)
ld_tm_val  = make_loader(X_tm_va,  y_tm_va,  batch_size, shuffle=False, drop_last=False)


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        torch.cuda.init()
        _ = torch.empty(1, device="cuda")
        torch.cuda.synchronize()

    return device

def PINN_train():
    # logs (kept local to avoid mixing with MLP logs)
    train_losses = []
    val_int_losses = []
    val_bc_losses = []
    val_term_losses = []
    total_val_losses = []
    pde_losses = []

    device = get_device()

    start = time.perf_counter()

    phi_mean_t = torch.tensor(phi_mean, dtype=torch.float32, device=device).view(1, 5)
    phi_std_t  = torch.tensor(phi_std,  dtype=torch.float32, device=device).view(1, 5)

    model = LocalVolatilityPINN(input_dim = 7, width=64, depth=8, activation="tanh").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    lam_int  = 1.0
    lam_bc   = 10.0
    lam_term = 10.0
    lam_pde  = 0.3

    epochs = 500
    best_val = float("inf")
    best_epoch = -1
    patience = 50
    min_delta = 1e-4
    wait = 0
    warmup = 75

    for epoch in range(1, epochs + 1):
        model.train()

        it_int = iter(ld_int)
        it_bc  = iter(ld_bc)
        it_tm  = iter(ld_tm)
        steps = min(len(ld_int), len(ld_bc), len(ld_tm))

        train_loss = 0.0
        train_parts = np.zeros(4, dtype=np.float64)  # [Lint, Lbc, Lterm, Lpde]

        for _ in range(steps):
            X_int, y_int = next(it_int)
            X_bc,  y_bc  = next(it_bc)
            X_tm,  y_tm  = next(it_tm)

            X_int = X_int.to(device); y_int = y_int.to(device)
            X_bc  = X_bc.to(device);  y_bc  = y_bc.to(device)
            X_tm  = X_tm.to(device);  y_tm  = y_tm.to(device)

            opt.zero_grad()

            X_int = X_int.detach().requires_grad_(True)

            #supervised losses
            pred_int = model(X_int)
            pred_bc  = model(X_bc)
            pred_tm  = model(X_tm)

            L_int  = mse(pred_int, y_int)
            L_bc   = mse(pred_bc,  y_bc)
            L_term = mse(pred_tm,  y_tm)

            R = Normalized_PDE_residual(
                model, X_int, K=K, r=r,
                S_min=S_min, S_max=S_max, t_min=t_min, t_max=t_max,
                phi_mean=phi_mean_t, phi_std=phi_std_t,
            )
            L_pde = torch.mean(R**2)

            loss = lam_int*L_int + lam_bc*L_bc + lam_term*L_term + lam_pde*L_pde
            loss.backward()
            opt.step()

            train_loss += float(loss.detach().cpu())
            train_parts += np.array([
                float(L_int.detach().cpu()),
                float(L_bc.detach().cpu()),
                float(L_term.detach().cpu()),
                float(L_pde.detach().cpu())
            ])

        train_loss /= steps
        train_parts /= steps

        #validation
        model.eval()
        with torch.no_grad():

            def eval_loader(loader):
                total = 0.0
                n = 0
                for Xb, yb in loader:
                    Xb = Xb.to(device); yb = yb.to(device)
                    total += float(mse(model(Xb), yb).detach().cpu())
                    n += 1
                return total / max(n, 1)

            val_int  = eval_loader(ld_int_val)
            val_bc   = eval_loader(ld_bc_val)
            val_term = eval_loader(ld_tm_val)

            val_total = (
                lam_int  * val_int +
                lam_bc   * val_bc +
                lam_term * val_term
            )

        print(
            f"Epoch {epoch:04d} | "
            f"Training loss: {train_loss:.3e} "
            f"Validation loss: {val_total:.3e}"
        )

        train_losses.append(train_loss)
        val_int_losses.append(val_int)
        val_bc_losses.append(val_bc)
        val_term_losses.append(val_term)
        total_val_losses.append(val_total)
        pde_losses.append(train_parts[3])

        #save best model based on validation loss
        improved = (best_val - val_total) > min_delta

        if improved:
            best_val = val_total
            best_epoch = epoch
            wait = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "stats": stats,
                    "K": K,
                    "r": r,
                },
                "PINN_model.pt"
            )
        else:
            if epoch >= warmup:
                wait += 1

        if epoch >= warmup and wait >= patience:
            print(
                f"Early stopping at epoch {epoch} with best epoch {best_epoch} "
                f"and best val {best_val:.3e}"
            )
            break

    np.savez_compressed(
        "PINN_training_logs.npz",
        train_losses=np.array(train_losses, dtype=np.float32),
        val_int=np.array(val_int_losses, dtype=np.float32),
        val_bc=np.array(val_bc_losses, dtype=np.float32),
        val_term=np.array(val_term_losses, dtype=np.float32),
        val_total=np.array(total_val_losses, dtype=np.float32),
        pde=np.array(pde_losses, dtype=np.float32),
    )

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    print(f"PINN training completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    PINN_train()