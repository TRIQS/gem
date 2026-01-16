import numpy as np
from triqs_ghostGA.utility.delta_fit import *

# --- Configuration ---
# Use a small size and beta=1.0 for debugging to avoid vanishing gradients
size = 2
p = 1
beta = 200.0 
n = size

def check_gradient_LR(x_test, beta, Lambda_c, D, F22_target, RTF12_target):
    """
    Compares the analytical Jacobian from new_test.py with a 
    manually computed numerical Jacobian using finite differences.
    """
    # 1. Calculate Analytical Jacobian
    print("Calculating analytical Jacobian...")
    ana_jac = jacobian_LR(x_test, beta, Lambda_c, D, F22_target, RTF12_target)
    
    # 2. Calculate Numerical Jacobian
    def wrapped_residual_LR(x):
        return residual_LR(x, beta, Lambda_c, D, F22_target, RTF12_target)
    
    print("Computing numerical Jacobian via finite differences...")
    f0 = wrapped_residual_LR(x_test)
    num_jac = []
    epsilon = 1e-9   # Optimal step size for double precision
    
    for i in range(len(x_test)):
        x_eps = np.copy(x_test)
        x_eps[i] += epsilon
        f1 = wrapped_residual_LR(x_eps)
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
    
    if max_diff < 1e-5:
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


def check_gradient_LcD(x_test, beta, Lambda, R, F11_target, F12D_target):
    """
    Compares the analytical Jacobian from new_test.py with a 
    manually computed numerical Jacobian using finite differences.
    """
    # 1. Calculate Analytical Jacobian
    print("Calculating analytical Jacobian...")
    ana_jac = jacobian_LcD(x_test, beta, Lambda, R, F11_target, F12D_target)
    
    # 2. Calculate Numerical Jacobian
    def wrapped_residual_LcD(x):
        return residual_LcD(x, beta, Lambda, R, F11_target, F12D_target)
    
    print("Computing numerical Jacobian via finite differences...")
    f0 = wrapped_residual_LcD(x_test)
    num_jac = []
    epsilon = 1e-9   # Optimal step size for double precision
    
    for i in range(len(x_test)):
        x_eps = np.copy(x_test)
        x_eps[i] += epsilon
        f1 = wrapped_residual_LcD(x_eps)
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
    
    if max_diff < 1e-5:
        print("SUCCESS: Analytical and Numerical Jacobians match.")
    else:
        print("FAILURE: Significant mismatch detected.")
        bad_row, bad_col = np.unravel_index(np.argmax(diff), diff.shape)
        
        # Identify the parameter type of the failing column
        if bad_col < n_re:
            p_type = f"Lambda_c Real (element {bad_col})"
        elif bad_col < n_re + n_im:
            p_type = f"Lambda_c Imag (element {bad_col - n_re})"
        elif bad_col < n_re + n_im + n*p:
            p_type = f"D Real (element {bad_col - (n_re + n_im)})"
        else:
            p_type = f"D Imag (element {bad_col - (n_re + n_im + n*p)})"
            
        print(f"Largest error at Row {bad_row}, Col {bad_col} ({p_type})")
        print(f"Analytical value: {ana_jac[bad_row, bad_col]:.6f}")
        print(f"Numerical value:  {num_jac[bad_row, bad_col]:.6f}")
    print("-" * 40)

    
if __name__ == "__main__":
    # --- Setup Test Case ---
    # Generate consistent random matrices
    np.random.seed(420) 
    
    D_test = np.random.randn(n, p) + 1j*np.random.randn(n, p)
    Lc_test = np.random.randn(n, n) + 1j*np.random.randn(n, n)
    Lc_test = 0.5 * (Lc_test + Lc_test.conj().T)
    
    # Random starting point for x
    L_test = np.random.randn(n, n) + 1j*np.random.randn(n, n)
    L_test = 0.5 * (L_test + L_test.conj().T)
    R_test = np.random.randn(n, p) + 1j*np.random.randn(n, p)
    
    x_start_LR  = pack_params(L_test, R_test)
    x_start_LcD = pack_params(Lc_test, D_test)

    
    H = build_H(L_test, Lc_test, D_test, R_test)
    Delta_trg = F_of_H(H, beta)
    

    F11_trg=Delta_trg[:n,:n]
    F22_trg=Delta_trg[n:,n:]
    F12_trg=Delta_trg[:n,n:]
    F12D_trg = F12_trg@D_test
    RTF12_trg = R_test.T@F12_trg

    print(" --- CHECK GRADIENT WITH RESPECT TO LAMBDA AND R ---")
    check_gradient_LR(x_start_LR, beta, Lc_test, D_test, F22_trg, RTF12_trg)

    print(" --- CHECK GRADIENT WITH RESPECT TO LAMBDA_C AND D ---")
    check_gradient_LcD(x_start_LcD, beta, L_test, R_test, F11_trg, F12D_trg)
