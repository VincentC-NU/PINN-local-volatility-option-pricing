import numpy as np
from Local_volatility_function import local_volatility, local_volatility_sampling
from Crank_Nicolson_PDE_solver import crank_nicolson_solver

#boundary and terminal condition helper functions
def payoff(S, K, option_type="call"):
    """
    Payoff function at maturity (t=0).
    """
    if option_type == "call":
        return np.maximum(S - K, 0.0)
    elif option_type == "put":
        return np.maximum(K - S, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def make_terminal_rows(S_grid, t0, Phi, K, option_type="call"):
    """
    Terminal condition at t=t0 (in your solver convention, t_grid[0] is terminal/maturity).
    """
    S = S_grid.astype(float)
    t = np.full_like(S, float(t0), dtype=float)
    y = payoff(S, K, option_type)

    Phi_block = np.repeat(Phi[None, :], len(S), axis=0)
    X = np.column_stack([S, t, Phi_block])
    return X, y.reshape(-1, 1)


def make_boundary_rows(S_min, S_max, t_grid, Phi, K, r, option_type="call"):
    """
    Spatial boundary conditions at S=S_min and S=S_max for all t in t_grid.
    """
    t = t_grid.astype(float)
    disc = np.exp(-r * t)

    #left boundary: S = S_min
    S_left = np.full_like(t, float(S_min), dtype=float)
    #right boundary: S = S_max
    S_right = np.full_like(t, float(S_max), dtype=float)

    if option_type == "call":
        y_left = np.zeros_like(t) #call option is worthless if S=0 (or very small)
        y_right = float(S_max) - K * disc #call option value approaches S-K*exp(-rt) as S->infinity
    else: 
        y_left = K * disc #put option value approaches K*exp(-rt) as S->0
        y_right = np.zeros_like(t) #put option is worthless if S is very large

    Phi_block = np.repeat(Phi[None, :], len(t), axis=0)

    X_left = np.column_stack([S_left, t, Phi_block])
    X_right = np.column_stack([S_right, t, Phi_block])

    X = np.vstack([X_left, X_right])
    y = np.concatenate([y_left, y_right]).reshape(-1, 1)
    return X, y

#main dataset generation
def main():
    K = 100.0
    r = 0.05
    option_type = "call"
    S_grid = np.linspace(1e-3, 400, 500)
    t_grid = np.linspace(0.0, 1.0, 500)

    #dataset sizes
    n_surfaces = 200 #number of different local volatility surfaces (i.e. different Phi) to sample
    n_points_per_surface = 20000  #interior supervised samples per surface (subsampled)

    #local volatility sampling bounds
    sigma_min = 0.05
    sigma_max = 1.0

    rng = np.random.default_rng(123)

    #per-surface storage (important for surface-level splitting)
    phis = []
    """
    200 volatility surfaces
    each surface has 20000 interior supervised points (S,t,Phi) -> V
    each surface has 1000 boundary condition points (S_min,t,Phi) and (S_max,t,Phi) -> V
    each surface has 500 terminal condition points (S,t0,Phi) -> payoff
    """
    X_int_list, y_int_list = [], []
    X_bc_list, y_bc_list = [], []
    X_term_list, y_term_list = [], []

    M, N = len(S_grid), len(t_grid)
    total_points = M * N

    for s in range(n_surfaces):
        #sample Phi
        Phi = local_volatility_sampling(
            S_grid, t_grid,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rng=rng
        )
        phis.append(Phi)

        #build sigma surface and solve PDE
        sigma_grid = local_volatility(S_grid, t_grid, Phi)
        V = crank_nicolson_solver(S_grid, t_grid, sigma_grid, K=K, r=r, option_type=option_type)

        #get interior supervised points (subsample from the grid)
        idx = rng.choice(total_points, size=n_points_per_surface, replace=False)
        i = idx // N
        j = idx % N

        S_samples = S_grid[i]
        t_samples = t_grid[j]
        Phi_samples = np.repeat(Phi[None, :], n_points_per_surface, axis=0)
        V_samples = V[i, j]

        X_int = np.column_stack([S_samples, t_samples, Phi_samples]) #(20000, 7)
        y_int = V_samples.reshape(-1, 1)  #(20000, 1)

        X_int_list.append(X_int)
        y_int_list.append(y_int)

        #terminal (payoff) points: all S at t=t_grid[0]
        X_term, y_term = make_terminal_rows(S_grid, t_grid[0], Phi, K, option_type=option_type)
        X_term_list.append(X_term)
        y_term_list.append(y_term)

        #boundary points: all t at S=S_min and S=S_max
        X_bc, y_bc = make_boundary_rows(S_grid[0], S_grid[-1], t_grid, Phi, K, r, option_type=option_type)
        X_bc_list.append(X_bc)
        y_bc_list.append(y_bc)

        if (s + 1) % 10 == 0:
            print(f"Done {s+1}/{n_surfaces} surfaces")

    phis = np.array(phis)  # (n_surfaces, 5)

    #surface-level train/val split
    val_ratio = 0.2
    surface_ids = np.arange(n_surfaces)
    rng.shuffle(surface_ids)
    n_val = int(val_ratio * n_surfaces)
    val_surfaces = set(surface_ids[:n_val])

    def split_by_surface(X_list, y_list):
        X_train, y_train, X_val, y_val = [], [], [], []
        for s in range(n_surfaces):
            if s in val_surfaces:
                X_val.append(X_list[s]); y_val.append(y_list[s])
            else:
                X_train.append(X_list[s]); y_train.append(y_list[s])
        return (np.vstack(X_train), np.vstack(y_train),
                np.vstack(X_val),   np.vstack(y_val))

    X_int_train, y_int_train, X_int_val, y_int_val = split_by_surface(X_int_list, y_int_list)
    X_bc_train,  y_bc_train,  X_bc_val,  y_bc_val  = split_by_surface(X_bc_list,  y_bc_list)
    X_term_train,y_term_train,X_term_val,y_term_val= split_by_surface(X_term_list,y_term_list)

    #convert to float32 for smaller files / faster training
    X_int_train  = X_int_train.astype(np.float32);  y_int_train  = y_int_train.astype(np.float32)
    X_int_val    = X_int_val.astype(np.float32);    y_int_val    = y_int_val.astype(np.float32)
    X_bc_train   = X_bc_train.astype(np.float32);   y_bc_train   = y_bc_train.astype(np.float32)
    X_bc_val     = X_bc_val.astype(np.float32);     y_bc_val     = y_bc_val.astype(np.float32)
    X_term_train = X_term_train.astype(np.float32); y_term_train = y_term_train.astype(np.float32)
    X_term_val   = X_term_val.astype(np.float32);   y_term_val   = y_term_val.astype(np.float32)
    phis         = phis.astype(np.float32)

    print("Interior train/val:", X_int_train.shape, X_int_val.shape)
    print("BC train/val:",       X_bc_train.shape,  X_bc_val.shape)
    print("Term train/val:",     X_term_train.shape,X_term_val.shape)
    print("phis:", phis.shape)

    #save dataset
    np.savez_compressed(
        "Localvol_dataset.npz",
        #interior supervised points
        X_int_train=X_int_train, y_int_train=y_int_train,
        X_int_val=X_int_val,     y_int_val=y_int_val,

        #boundary condition points
        X_bc_train=X_bc_train,   y_bc_train=y_bc_train,
        X_bc_val=X_bc_val,       y_bc_val=y_bc_val,

        #terminal condition points
        X_term_train=X_term_train, y_term_train=y_term_train,
        X_term_val=X_term_val,     y_term_val=y_term_val,

        #metadata
        phis=phis,
        S_grid=S_grid.astype(np.float32),
        t_grid=t_grid.astype(np.float32),
        K=np.array([K], dtype=np.float32),
        r=np.array([r], dtype=np.float32),
        option_type=np.array([option_type])
    )
    print("Saved: Localvol_dataset_train.npz")

if __name__ == "__main__":
    main()