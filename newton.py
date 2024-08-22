def diff(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)
    
def diffdiff(f, x, h=1e-6):
    return (diff(f, x + h, h) - diff(f, x - h, h)) / (2 * h)
    
def optimize(f, x0, tol=1e-6, max_iter=1000):
    x = x0
    for i in range(max_iter):
        f_diff = diff(f, x)
        f_diffdiff = diffdiff(f, x)
        
        if f_diffdiff == 0:
            print("Second derivative is zero. Stopping iteration.")
            return x, f(x), i
        
        x_new = x - f_diff / f_diffdiff
        if abs(x_new - x) < tol:
            return x_new, f(x_new), i + 1
        x = x_new
    print("Maximum iterations reached.")
    return x, f(x), max_iter