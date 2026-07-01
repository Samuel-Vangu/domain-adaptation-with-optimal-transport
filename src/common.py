import numpy as np
from ot import sinkhorn


class GCGSolver:
    """
    Generalized Conditional Gradient solver.
    
    User only needs to define the objective function f(γ, C).
    The gradient is computed automatically via finite differences.
    """
    
    def __init__(self, max_iter=1000, tol=1e-6, reg_e=0.1, verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.reg_e = reg_e
        self.verbose = verbose
        self.coupling_ = None
        self.cost_history_ = []
    
    def _compute_gradient(self, gamma, C, objective_func, eps=1e-5):
        """
        Compute gradient of objective function using finite differences.
        
        ∂f/∂γ_ij ≈ (f(γ + eps·E_ij) - f(γ)) / eps
        
        where E_ij is a matrix with 1 at position (i,j) and 0 elsewhere.
        """
        n_s, n_t = gamma.shape
        gradient = np.zeros_like(gamma)
        base_cost = objective_func(gamma, C)
        
        for i in range(n_s):
            for j in range(n_t):
                # Perturb gamma[i, j]
                gamma_perturbed = gamma.copy()
                gamma_perturbed[i, j] += eps
                
                # Compute partial derivative
                perturbed_cost = objective_func(gamma_perturbed, C)
                gradient[i, j] = (perturbed_cost - base_cost) / eps
        
        return gradient
    
    def _line_search(self, gamma, delta_gamma, C, objective_func, n_steps=20):
        """Find optimal step size α ∈ [0, 1] via grid search."""
        alphas = np.linspace(0, 1, n_steps)
        best_alpha = 0.0
        best_cost = float('inf')
        
        for alpha in alphas:
            gamma_candidate = np.maximum(gamma + alpha * delta_gamma, 1e-10)
            
            # Total cost = objective + entropy regularization
            cost_f = objective_func(gamma_candidate, C)
            cost_entropy = -self.reg_e * np.sum(
                gamma_candidate * np.log(gamma_candidate + 1e-10)
            )
            total_cost = cost_f + cost_entropy
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_alpha = alpha
        
        return best_alpha
    
    def fit(self, C, a, b, objective_func, gamma_init=None):
        """
        Solve the regularized OT problem.
        
        Parameters
        ----------
        C : np.ndarray (n_s, n_t)
            Cost matrix
        a : np.ndarray (n_s,)
            Source distribution
        b : np.ndarray (n_t,)
            Target distribution
        objective_func : Callable
            YOUR objective function f(γ, C) -> float
            Example: f(γ, C) = <γ, C> + η·Ω_class(γ)
        gamma_init : np.ndarray, optional
            Initial transport plan
        
        Returns
        -------
        self
        """
        n_s, n_t = C.shape
        
        # Initialize
        if gamma_init is None:
            gamma = np.outer(a, b)
        else:
            gamma = gamma_init.copy()
        
        gamma = np.maximum(gamma, 1e-10)
        self.cost_history_ = []
        
        # GCG iterations
        for k in range(self.max_iter):
            
            # Step 1: Compute gradient (AUTOMATIC via finite differences)
            G = self._compute_gradient(gamma, C, objective_func)
            
            # Step 2: Solve entropy-regularized subproblem with Sinkhorn
            gamma_star = sinkhorn(a, b, G, reg=self.reg_e, numItermax=1000)
            
            # Step 3: Line search
            delta_gamma = gamma_star - gamma
            alpha = self._line_search(gamma, delta_gamma, C, objective_func)
            
            # Step 4: Update
            gamma_new = np.maximum(gamma + alpha * delta_gamma, 1e-10)
            
            # Monitor convergence
            current_cost = objective_func(gamma_new, C)
            self.cost_history_.append(current_cost)
            
            max_change = np.max(np.abs(gamma_new - gamma))
            
            if self.verbose:
                print(f"Iter {k+1:4d}: cost={current_cost:.6f}, "
                      f"α={alpha:.4f}, Δ={max_change:.2e}")
            
            if max_change < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {k+1}")
                break
            
            gamma = gamma_new
        
        self.coupling_ = gamma
        return self

def cost(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    # vec.shape = (d,)
    # mat.shape = (n, m) 
    # return a ndarray of shape (k,)
    return np.linalg.norm(mat - vec , axis=1)


def cost_matrix(A: np.ndarray, B: np.ndarray):
    """Return the cost matrix between two datasets"""
    # A.shape = (N, d)   → N vectors
    # B.shape = (n, m)
    
    def wrapper(row):
        return cost(row, B)
    
    C = np.apply_along_axis(wrapper, axis=1, arr=A)
    return C