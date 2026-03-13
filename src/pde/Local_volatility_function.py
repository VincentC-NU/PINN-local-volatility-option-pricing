import numpy as np

#S: underlying asset price
#t: time to maturity (normalized to [0,1])
#K: strike price (set to 100)
#Phi: parameters of the local volatility function [sigma_0, alpha, beta, gamma, delta]
#sigma_0: base volatility level
#alpha: skew strength
#beta: skew steepness
#gamma: term structure amplitude
#eta: term structure decay rate

K = 100.0

def local_volatility(S, t, Phi): #local volatility function
    S = np.asarray(S)
    t = np.asarray(t)
    Phi = np.asarray(Phi)

    sigma0, alpha, beta, gamma, eta = Phi
    S_safe = np.maximum(S, 1e-8)
    
    log_m = np.log(S_safe / K)
    skew = 1.0 + alpha * np.tanh(beta * log_m)
    term = 1.0 + gamma * np.exp(-eta * t)
    sigma_val = sigma0 * skew[:, None] * term[None, :]

    if np.min(sigma_val) <= 0:
        raise ValueError("Negative volatility encountered.")

    return sigma_val

def local_volatility_sampling(S_grid, t_grid, sigma_min = 0.05, sigma_max = 1.00, rng = None):
    if rng is None:
        rng = np.random.default_rng()

    S_grid = np.asarray(S_grid)
    t_grid = np.asarray(t_grid)
    
    # Parameter ranges for sampling
    max_tries = 1000
    param_ranges = {
        "sigma0": (0.12, 0.35),
        "alpha":  (-0.6, 0.6),
        "beta":   (0.5, 3.0),
        "gamma":  (-0.4, 0.8),
        "eta":    (0.5, 3.0),
        }

    for _ in range(max_tries):
        sigma0 = rng.uniform(*param_ranges["sigma0"])
        alpha  = rng.uniform(*param_ranges["alpha"])
        beta   = rng.uniform(*param_ranges["beta"])
        gamma  = rng.uniform(*param_ranges["gamma"])
        eta    = rng.uniform(*param_ranges["eta"])

        Phi = np.array([sigma0, alpha, beta, gamma, eta], dtype=float) #create oaraneter vector

        # Compute sigma surface on your grid and check bounds
        sig = local_volatility(S_grid, t_grid, Phi)
        smin = float(sig.min())
        smax = float(sig.max())

        if (smin >= sigma_min) and (smax <= sigma_max):
            return Phi

    raise RuntimeError(
        f"Failed to sample Phi in {max_tries} tries. "
        f"Try widening sigma_min/sigma_max or tightening parameter ranges."
    )