import numpy as np
from delta_fit import *

# --- Configuration ---
# Use a small size and beta=1.0 for debugging to avoid vanishing gradients
size = 2
p = 1
beta = 1.0 
n = size

def check_gradient(x_test, beta, Lambda_c, D, F11_target, F12_target):
    """
    Compares the analytical Jacobian from new_test.py with a 
    manually computed numerical Jacobian using finite differences.
    """
    # 1. Calculate Analytical Jacobian
    print("Calculating analytical Jacobian...")
    ana_jac = jacobian(x_test, beta, Lambda_c, D, F11_target, F12_target)
    
    # 2. Calculate Numerical Jacobian
    def wrapped_residual(x):
        return residual(x, beta, Lambda_c, D, F11_target, F12_target)
    
    print("Computing numerical Jacobian via finite differences...")
    f0 = wrapped_residual(x_test)
    num_jac = []
    epsilon = 1e-7  # Optimal step size for double precision
    
    for i in range(len(x_test)):
        x_eps = np.copy(x_test)
        x_eps[i] += epsilon
        f1 = wrapped_residual(x_eps)
        # Central difference could be used for even higher precision, 
        # but forward difference (shown here) is standard for comparison.
        num_jac.append((f1 - f0) / epsilon)
    
    num_jac = np.column_stack(num_jac)
    
    # 3. Comprehensive Comparison
    diff = np.abs(ana_jac - num_jac)
    max_diff = np.max(diff)
    
    # Determine which part of the vector x failed
    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)
    n_re = len(iu[0])
    n_im = len(iu_strict[0])
    
    print("-" * 40)
    print(f"Jacobian Shape: {ana_jac.shape}")
    print(f"Max Absolute Difference: {max_diff:.2e}")
    
    if max_diff < 1e-6:
        print("SUCCESS: Analytical and Numerical Jacobians match.")
    else:
        print("FAILURE: Significant mismatch detected.")
        bad_row, bad_col = np.unravel_index(np.argmax(diff), diff.shape)
        
        # Identify the parameter type of the failing column
        if bad_col < n_re:
            p_type = f"Lambda Real (element {bad_col})"
        elif bad_col < n_re + n_im:
            p_type = f"Lambda Imag (element {bad_col - n_re})"
        elif bad_col < n_re + n_im + n*p:
            p_type = f"R Real (element {bad_col - (n_re + n_im)})"
        else:
            p_type = f"R Imag (element {bad_col - (n_re + n_im + n*p)})"
            
        print(f"Largest error at Row {bad_row}, Col {bad_col} ({p_type})")
        print(f"Analytical value: {ana_jac[bad_row, bad_col]:.6f}")
        print(f"Numerical value:  {num_jac[bad_row, bad_col]:.6f}")
    print("-" * 40)

if __name__ == "__main__":
    # --- Setup Test Case ---
    # Generate consistent random matrices
    np.random.seed(42) 
    
    D_test = np.random.randn(n, p) + 1j*np.random.randn(n, p)
    Lc_test = np.random.randn(n, n) + 1j*np.random.randn(n, n)
    Lc_test = 0.5 * (Lc_test + Lc_test.conj().T)
    
    # Random starting point for x
    L_start = np.random.randn(n, n) + 1j*np.random.randn(n, n)
    L_start = 0.5 * (L_start + L_start.conj().T)
    R_start = np.random.randn(n, p) + 1j*np.random.randn(n, p)
    
    x_start = pack_params(L_start, R_start)
    
    # Targets (zeroed out as they don't affect the Jacobian matrix itself, only the residual value)
    F11_t = np.zeros((n, n), dtype=complex)
    F12_t = np.zeros((n, p), dtype=complex)

    check_gradient(x_start, beta, Lc_test, D_test, F11_t, F12_t)
