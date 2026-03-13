import numpy as np
from Local_volatility_function import local_volatility, local_volatility_sampling
from Crank_Nicolson_PDE_solver import crank_nicolson_solver
from Data_train import make_terminal_rows, make_boundary_rows


def main():
    K = 100.0
    r = 0.05
    option_type = "call"

    S_grid = np.linspace(1e-3, 400, 500)
    t_grid = np.linspace(0.0, 1.0, 500)

    n_test = 40               
    n_points_per_surface = 20000

    sigma_min = 0.05
    sigma_max = 1.0

    rng = np.random.default_rng(456)

    phis = []

    X_int_list, y_int_list = [], []
    X_bc_list, y_bc_list = [], []
    X_term_list, y_term_list = [], []

    M, N = len(S_grid), len(t_grid)
    total_points = M * N
    if n_points_per_surface > total_points:
        raise ValueError(
            f"n_points_per_surface={n_points_per_surface} exceeds grid points={total_points}"
        )

    for s in range(n_test):
        Phi = local_volatility_sampling(
            S_grid, t_grid,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rng=rng
        )
        phis.append(Phi)

        sigma_grid = local_volatility(S_grid, t_grid, Phi)
        V = crank_nicolson_solver(
            S_grid, t_grid, sigma_grid,
            K=K, r=r, option_type=option_type
        )

        idx = rng.choice(total_points, size=n_points_per_surface, replace=False)
        i = idx // N
        j = idx % N

        S_samples = S_grid[i]
        t_samples = t_grid[j]
        Phi_samples = np.repeat(Phi[None, :], n_points_per_surface, axis=0)
        V_samples = V[i, j]

        X_int = np.column_stack([S_samples, t_samples, Phi_samples])
        y_int = V_samples.reshape(-1, 1)                               

        X_int_list.append(X_int)
        y_int_list.append(y_int)

        X_term, y_term = make_terminal_rows(S_grid, t_grid[0], Phi, K, option_type=option_type)
        X_term_list.append(X_term)
        y_term_list.append(y_term)

        X_bc, y_bc = make_boundary_rows(S_grid[0], S_grid[-1], t_grid, Phi, K, r, option_type=option_type)
        X_bc_list.append(X_bc)
        y_bc_list.append(y_bc)

    phis = np.array(phis) 

    X_int = np.concatenate(X_int_list, axis=0)
    y_int = np.concatenate(y_int_list, axis=0)
    X_bc = np.concatenate(X_bc_list, axis=0)
    y_bc = np.concatenate(y_bc_list, axis=0)
    X_term = np.concatenate(X_term_list, axis=0)
    y_term = np.concatenate(y_term_list, axis=0)

    np.savez(
        "PINN_test_data.npz",
        X_int=X_int, y_int=y_int,
        X_bc=X_bc, y_bc=y_bc,
        X_term=X_term, y_term=y_term,
        phis=phis,
        K=K, r=r,
        S_min=S_grid[0], S_max=S_grid[-1],
        t_min=t_grid[0], t_max=t_grid[-1],
        S_grid=S_grid, t_grid=t_grid,
        option_type=option_type,
        sigma_min=sigma_min, sigma_max=sigma_max,
        seed=456,
        n_test=n_test,
        n_points_per_surface=n_points_per_surface,
        phi_mean=np.mean(phis, axis=0),
        phi_std=np.std(phis, axis=0),
    )

    print("Saved: PINN_test_data.npz")


if __name__ == "__main__":
    main()