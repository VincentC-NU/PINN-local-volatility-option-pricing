import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader

# DataLoader
def make_loader(X, y, batch_size, shuffle=True, drop_last=True):
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

#symmetric max-min normalization
def Normalization(x, xmin, xmax, eps=1e-12):
    return 2.0 * (x - xmin) / (xmax - xmin + eps) - 1.0

def Gaussian_noise(y, sigma=0.01, seed = 42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=y.shape)
    return y + noise

def prepare_normalization(X_int_tr, K=None, r=None):
    S_min = float(X_int_tr[:, 0].min())
    S_max = float(X_int_tr[:, 0].max())
    t_min = float(X_int_tr[:, 1].min())
    t_max = float(X_int_tr[:, 1].max())

    phi_train = X_int_tr[:, 2:7]
    phi_mean = phi_train.mean(axis=0)
    phi_std  = phi_train.std(axis=0) + 1e-8

    stats = {
        "S_min": S_min, "S_max": S_max,
        "t_min": t_min, "t_max": t_max,
        "phi_mean": phi_mean, "phi_std": phi_std,
    }
    if K is not None: stats["K"] = float(K)
    if r is not None: stats["r"] = float(r)
    return stats

def apply_normalization(X, stats):
    Xn = X.copy()
    Xn[:, 0] = Normalization(Xn[:, 0], stats["S_min"], stats["S_max"])
    Xn[:, 1] = Normalization(Xn[:, 1], stats["t_min"], stats["t_max"])
    Xn[:, 2:7] = (Xn[:, 2:7] - stats["phi_mean"]) / stats["phi_std"]
    return Xn

if __name__ == "__main__":
    raw_data = np.load("Localvol_dataset_train.npz")

    X_int_tr, y_int_tr = raw_data["X_int_train"], raw_data["y_int_train"]
    X_bc_tr,  y_bc_tr  = raw_data["X_bc_train"],  raw_data["y_bc_train"]
    X_tm_tr,  y_tm_tr  = raw_data["X_term_train"],raw_data["y_term_train"]

    X_int_va, y_int_va = raw_data["X_int_val"], raw_data["y_int_val"]
    X_bc_va,  y_bc_va  = raw_data["X_bc_val"],  raw_data["y_bc_val"]
    X_tm_va,  y_tm_va  = raw_data["X_term_val"],raw_data["y_term_val"]

    K = float(raw_data["K"][0])
    r = float(raw_data["r"][0])
        
    #normalize y by K to stabilize training
    y_int_tr = y_int_tr / K
    y_bc_tr  = y_bc_tr  / K
    y_tm_tr  = y_tm_tr  / K
    
    y_int_va = y_int_va / K
    y_bc_va  = y_bc_va  / K
    y_tm_va  = y_tm_va  / K

    #compute normalization stats from training interior
    stats = prepare_normalization(X_int_tr)

    #apply normalization to ALL X sets
    X_int_tr_n = apply_normalization(X_int_tr, stats)
    X_bc_tr_n  = apply_normalization(X_bc_tr,  stats)
    X_tm_tr_n  = apply_normalization(X_tm_tr,  stats)

    X_int_va_n = apply_normalization(X_int_va, stats)
    X_bc_va_n  = apply_normalization(X_bc_va,  stats)
    X_tm_va_n  = apply_normalization(X_tm_va,  stats)

    #create DataLoaders
    batch_size = 4096
    ld_int = make_loader(X_int_tr_n, y_int_tr, batch_size, shuffle=True, drop_last=True)
    ld_bc  = make_loader(X_bc_tr_n,  y_bc_tr,  batch_size, shuffle=True, drop_last=True)
    ld_tm  = make_loader(X_tm_tr_n,  y_tm_tr,  batch_size, shuffle=True, drop_last=True)

    ld_int_val = make_loader(X_int_va_n, y_int_va, batch_size, shuffle=False, drop_last=False)
    ld_bc_val  = make_loader(X_bc_va_n,  y_bc_va,  batch_size, shuffle=False, drop_last=False)
    ld_tm_val  = make_loader(X_tm_va_n,  y_tm_va,  batch_size, shuffle=False, drop_last=False)

    #move normalization params needed by PDE residual to torch tensors
    S_min = stats["S_min"]; S_max = stats["S_max"]
    t_min = stats["t_min"]; t_max = stats["t_max"]
    phi_mean = stats["phi_mean"]; phi_std = stats["phi_std"]
    
    phi_mean_t = torch.tensor(phi_mean, dtype=torch.float32).view(1, 5)
    phi_std_t  = torch.tensor(phi_std,  dtype=torch.float32).view(1, 5)

    np.savez_compressed( #clean training dataset (tune hyperparameters)
        "Localvol_dataset_train_normalized_clean.npz",

        X_int_train=X_int_tr_n.astype(np.float32), y_int_train=y_int_tr.astype(np.float32),
        X_int_val  =X_int_va_n.astype(np.float32), y_int_val  =y_int_va.astype(np.float32),

        X_bc_train=X_bc_tr_n.astype(np.float32),   y_bc_train=y_bc_tr.astype(np.float32),
        X_bc_val  =X_bc_va_n.astype(np.float32),   y_bc_val  =y_bc_va.astype(np.float32),

        X_term_train=X_tm_tr_n.astype(np.float32), y_term_train=y_tm_tr.astype(np.float32),
        X_term_val  =X_tm_va_n.astype(np.float32), y_term_val  =y_tm_va.astype(np.float32),

        phis=raw_data["phis"],
        S_grid=raw_data["S_grid"].astype(np.float32),
        t_grid=raw_data["t_grid"].astype(np.float32),
        K=raw_data["K"].astype(np.float32),
        r=raw_data["r"].astype(np.float32),
        option_type=raw_data["option_type"],

        S_min=np.array([S_min], dtype=np.float32),
        S_max=np.array([S_max], dtype=np.float32),
        t_min=np.array([t_min], dtype=np.float32),
        t_max=np.array([t_max], dtype=np.float32),
        phi_mean=phi_mean.astype(np.float32),
        phi_std=phi_std.astype(np.float32),

        y_scaled_by=np.array(["K"], dtype=object),
    )
    print("Saved: Localvol_dataset_train_normalized_clean.npz")
    
    noise_sigma = 0.01
    y_int_tr_noisy = Gaussian_noise(y_int_tr, sigma=noise_sigma)
    
    np.savez_compressed( #noisy training datast (train models)
        "Localvol_dataset_train_normalized_noisy.npz",

        X_int_train=X_int_tr_n.astype(np.float32), y_int_train=y_int_tr_noisy.astype(np.float32),
        X_int_val  =X_int_va_n.astype(np.float32), y_int_val  =y_int_va.astype(np.float32),

        X_bc_train=X_bc_tr_n.astype(np.float32),   y_bc_train=y_bc_tr.astype(np.float32),
        X_bc_val  =X_bc_va_n.astype(np.float32),   y_bc_val  =y_bc_va.astype(np.float32),

        X_term_train=X_tm_tr_n.astype(np.float32), y_term_train=y_tm_tr.astype(np.float32),
        X_term_val  =X_tm_va_n.astype(np.float32), y_term_val  =y_tm_va.astype(np.float32),

        phis=raw_data["phis"],
        S_grid=raw_data["S_grid"].astype(np.float32),
        t_grid=raw_data["t_grid"].astype(np.float32),
        K=raw_data["K"].astype(np.float32),
        r=raw_data["r"].astype(np.float32),
        option_type=raw_data["option_type"],

        S_min=np.array([S_min], dtype=np.float32),
        S_max=np.array([S_max], dtype=np.float32),
        t_min=np.array([t_min], dtype=np.float32),
        t_max=np.array([t_max], dtype=np.float32),
        phi_mean=phi_mean.astype(np.float32),
        phi_std=phi_std.astype(np.float32),

        y_scaled_by=np.array(["K"], dtype=object),
    )
    print("Saved: Localvol_dataset_train_normalized_noisy.npz")
    
    # Normalize TEST dataset using the same stats as training
    test_data = np.load("Localvol_dataset_test.npz")

    X_int_te, y_int_te = test_data["X_int"],  test_data["y_int"]
    X_bc_te,  y_bc_te  = test_data["X_bc"],   test_data["y_bc"]
    X_tm_te,  y_tm_te  = test_data["X_term"], test_data["y_term"]

    # use training K for scaling (same as training)
    y_int_te = y_int_te / K
    y_bc_te  = y_bc_te  / K
    y_tm_te  = y_tm_te  / K

    # apply SAME normalization stats as training
    X_int_te_n = apply_normalization(X_int_te, stats)
    X_bc_te_n  = apply_normalization(X_bc_te,  stats)
    X_tm_te_n  = apply_normalization(X_tm_te,  stats)

    np.savez_compressed( #test dataset
        "Localvol_dataset_test_normalized.npz",

        X_int_test=X_int_te_n.astype(np.float32),
        y_int_test=y_int_te.astype(np.float32),

        X_bc_test=X_bc_te_n.astype(np.float32),
        y_bc_test=y_bc_te.astype(np.float32),

        X_term_test=X_tm_te_n.astype(np.float32),
        y_term_test=y_tm_te.astype(np.float32),

        phis=test_data["phis"],
        S_grid=test_data["S_grid"].astype(np.float32),
        t_grid=test_data["t_grid"].astype(np.float32),
        K=test_data["K"].astype(np.float32),
        r=test_data["r"].astype(np.float32),
        option_type=test_data["option_type"],

        S_min=np.array([S_min], dtype=np.float32),
        S_max=np.array([S_max], dtype=np.float32),
        t_min=np.array([t_min], dtype=np.float32),
        t_max=np.array([t_max], dtype=np.float32),
        phi_mean=phi_mean.astype(np.float32),
        phi_std=phi_std.astype(np.float32),

        y_scaled_by=np.array(["K"], dtype=object),
    )

    print("Saved: Localvol_dataset_test_normalized.npz")