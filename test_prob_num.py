import numpy as np
def compute_p_section32(eps, eps_prime):
    """
    Section 3.2 公式（eps2 = 0 的特殊情况）：
    p = (e^{ε'} - 1) / [(e^{ε/2} - 1) * (e^{ε' - ε/2} + 1)]
    """
    numerator = np.exp(eps_prime) - 1
    denominator = (np.exp(eps / 2) - 1) * (np.exp(eps_prime - eps / 2) + 1)
    p = numerator / denominator
    return np.clip(p, 0.0, 1.0)


print(compute_p_section32(20*2.89, 12*2.89))