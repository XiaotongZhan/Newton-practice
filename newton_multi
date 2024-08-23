import autograd.numpy as np
from autograd import grad, hessian

def optimize(f, x0, epsilon=1e-6, max_iter=1000):
    grad_f = grad(f)  #grad
    hess_f = hessian(f)  # Hessian
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        grad_val = grad_f(x)
        hess_val = hess_f(x)
        
        try:
            delta_x = np.linalg.solve(hess_val, grad_val)
        except np.linalg.LinAlgError:
            print("Hessian is singular and cannot be inverted.")
            break
        
        x_new = x - delta_x
        
        if np.linalg.norm(x_new - x) < epsilon:
            return x_new, f(x_new), i + 1
        
        x = x_new
    
    return x, f(x), max_iter


