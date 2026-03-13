import numpy as np 
from Local_volatility_function import local_volatility, local_volatility_sampling

def Thomas_solve(a, b, c, d):
    """
    Solves the tridiagonal system Ax = d where A has sub-diagonal a, diagonal b, and super-diagonal c.
    a: (n-1,) sub-diagonal entries
    b: (n,) diagonal entries
    c: (n-1,) super-diagonal entries
    d: (n,) right-hand vector
    """
    n = b.size 
    ac = a.astype(float).copy()
    bc = b.astype(float).copy()
    cc = c.astype(float).copy()
    dc = d.astype(float).copy()

    for i in range(1, n):
        w = ac[i-1] / bc[i-1]
        bc[i] -= w * cc[i-1]
        dc[i] -= w * dc[i-1]

    #Back substitution
    x = np.empty(n, dtype=float)
    x[-1] = dc[-1] / bc[-1] 
    for i in range(n-2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i+1]) / bc[i]
    return x


#solvinig Black-Scholes PDE with Crank-Nicolson method
def crank_nicolson_solver(S_grid, t_grid, sigma_grid, K, r, option_type="call"): 
    S = np.asarray(S_grid, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    sigma = np.asarray(sigma_grid, dtype=float)

    M = S.size
    N = t.size
    dS = S[1] - S[0]
    dt = t[1] - t[0]

    V = np.zeros((M, N), dtype=float)

    #initial condition at maturity t=0
    if option_type == "call":
        V[:, 0] = np.maximum(S - K, 0.0)
    elif option_type == "put":
        V[:, 0] = np.maximum(K - S, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    #boundary conditions for all times
    #left boundary i=0 at S≈0
    if option_type == "call":
        V[0, :] = 0.0
        V[-1, :] = S[-1] - K * np.exp(-r * t)
    else:
        V[0, :] = K * np.exp(-r * t)
        V[-1, :] = 0.0

    #interior indices
    i0, i1 = 1, M - 2
    m_int = M - 2
    S_int = S[i0:i1+1]

    #march forward in time-to-maturity: j -> j+1
    for j in range(0, N - 1):
        tj = t[j]
        tj1 = t[j+1]

        #use sigma at j and j+1 (CN average); both are fine, this is standard
        sig_j  = sigma[i0:i1+1, j]
        sig_j1 = sigma[i0:i1+1, j+1]

        #build L coefficients at time j
        a_j = 0.5 * (sig_j**2) * (S_int**2)
        b_j = r * S_int
        c0  = -r

        lj = a_j/dS**2 - b_j/(2*dS)
        dj = -2*a_j/dS**2 + c0
        uj = a_j/dS**2 + b_j/(2*dS)

        #build L coefficients at time j+1
        a_p = 0.5 * (sig_j1**2) * (S_int**2)
        b_p = r * S_int

        lp = a_p/dS**2 - b_p/(2*dS)
        dp = -2*a_p/dS**2 + c0
        up = a_p/dS**2 + b_p/(2*dS)

        #A = I - (dt/2) L_{j+1}
        A_lower = -(dt/2) * lp[1:]       # (m_int-1,)
        A_diag  = 1.0 - (dt/2) * dp      # (m_int,)
        A_upper = -(dt/2) * up[:-1]      # (m_int-1,)

        #B = I + (dt/2) L_{j}
        B_lower = +(dt/2) * lj[1:]
        B_diag  = 1.0 + (dt/2) * dj
        B_upper = +(dt/2) * uj[:-1]

        Vj_int = V[i0:i1+1, j]

        #RHS = B * V^j + boundary terms (CN average)
        rhs = B_diag * Vj_int
        rhs[1:]  += B_lower * Vj_int[:-1]
        rhs[:-1] += B_upper * Vj_int[1:]

        #boundary values at times j and j+1
        V0_j,  V0_j1  = V[0, j],  V[0, j+1]
        VM_j,  VM_j1  = V[-1, j], V[-1, j+1]

        #first interior node touches left boundary through l*
        rhs[0]  += (dt/2) * (lj[0] * V0_j  + lp[0] * V0_j1)
        #last interior node touches right boundary through u*
        rhs[-1] += (dt/2) * (uj[-1] * VM_j + up[-1] * VM_j1)

        #solve tridiagonal for V^{j+1}_interior
        V_next_int = Thomas_solve(A_lower, A_diag, A_upper, rhs)

        V[i0:i1+1, j+1] = V_next_int

    return V  #shape (M, N)