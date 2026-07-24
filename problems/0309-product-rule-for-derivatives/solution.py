import numpy as np

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' as a list of floats rounded to 4 decimal places
    """
        
    f_rev = f_coeffs[::-1]
    g_rev = g_coeffs[::-1]
    
    prod_rev = np.polymul(f_rev, g_rev)
    
    # Calculate the derivative: (f(x) * g(x))'
    deriv_rev = np.polyder(prod_rev)
    
    # Convert back to standard list layout (lowest power first)
    result = list(deriv_rev)[::-1]
    
    # Convert to floats and round to 4 decimal places
    result = [round(float(c), 4) for c in result]
    
    # Remove trailing zero coefficients from highest powers at the end
    while len(result) > 1 and result[-1] == 0.0:
        result.pop()
        
    # Ensure a completely zero polynomial returns [0.0]
    if not result:
        return [0.0]
        
    return result
