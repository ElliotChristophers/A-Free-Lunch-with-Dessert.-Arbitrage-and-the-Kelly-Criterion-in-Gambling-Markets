import numpy as np
import sympy as sp
import time

def generate_equations_for_n(n):
    f = sp.symbols(f'f1:{n+1}')
    b = sp.symbols(f'b1:{n+1}')
    p = sp.symbols(f'p1:{n+1}')
    equations = []
    for i in range(n):
        equation = 0
        for j in range(n):
            if j == i:
                equation += p[j] * b[j] / (1 + f[j] * b[j] - sum(f[:j]) - sum(f[j+1:n]))
            else:
                equation -= p[j] / (1 - sum(f[:j+1]) + f[j] * b[j] - sum(f[j+1:n])+f[j])
        equations.append(equation)
    return equations


def evaluate_equations(n, equations, values):
    f = sp.symbols(f'f1:{n+1}')
    b = sp.symbols(f'b1:{n+1}')
    p = sp.symbols(f'p1:{n+1}')

    substitution_dict = {**dict(zip(f, values['f'])), **dict(zip(b, values['b'])), **dict(zip(p, values['p']))}
    return [round(eq.subs(substitution_dict),10) for eq in equations]

l = []
for n in range(2,31):
    start = time.time()
    equations = generate_equations_for_n(n)
    vals = np.random.rand(n)
    scaled_values = vals * np.random.uniform(0.95, 1) / np.sum(vals)
    values = {}
    values['b'] = 1/(1+scaled_values)
    values['p'] = (1/(1+values['b']))/np.sum(1/(1+values['b']))
    values['f'] = np.ones(n)/n

    evaluated_results = evaluate_equations(n,equations,values)
    
    l.append([n, all(result == evaluated_results[0] for result in evaluated_results), time.time() - start])
    print(n,l[-1][-1])
