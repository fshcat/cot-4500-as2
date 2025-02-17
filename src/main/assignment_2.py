import numpy as np

def neville_method(xs, ys, x):
    n = len(xs)
    Q = np.zeros((n, n))

    for i in range(n):
        Q[i, 0] = ys[i]
    
    for j in range(1, n):
        for i in range(j, n):
            Q[i, j] = ((x - xs[i - j]) * Q[i, j - 1] - (x - xs[i]) * Q[i - 1, j - 1]) / (xs[i] - xs[i - j])

    return Q[n-1, n-1]

def newton_forward_difference(xs, ys):
    n = len(xs)
    diff_table = np.zeros((n, n))

    for i in range(n):
        diff_table[i, 0] = ys[i]

    for j in range(1, n):
        for i in range(j, n):
            diff_table[i, j] = diff_table[i, j - 1] - diff_table[i - 1, j - 1]
            diff_table[i, j] /= xs[i] - xs[i - j]       

    return diff_table

def newton_forward_interpolation(xs, ys, x):
    diff_table = newton_forward_difference(xs, ys)
    n = len(diff_table)

    term = 1
    result = ys[0]

    for i in range(1, n):
        term *= (x - xs[i - 1])
        result += diff_table[i, i] * term

    return result

def hermite_forward_difference(xs, ys, derivs):
    n = len(xs)

    diff_table = np.zeros((2 * n, n+1))

    for i in range(n):
        diff_table[2 * i, 0] = ys[i]
        diff_table[2 * i + 1, 0] = ys[i]

        diff_table[2 * i + 1, 1] = derivs[i]

    for i in range(1, n):
        diff_table[2*i, 1] = diff_table[2*i, 0] - diff_table[2*i - 1, 0]
        diff_table[2*i, 1] /= xs[i] - xs[i - 1]

    for j in range(2, n+1):
        for i in range(j, 2*n):
            diff_table[i, j] = diff_table[i, j - 1] - diff_table[i - 1, j - 1]
            diff_table[i, j] /= xs[i // 2] - xs[(i - j) // 2]

    new_table = np.zeros((2*n, n+2))
    new_table[:, 1:] = diff_table
    new_table[:, 0][::2] = xs
    new_table[:, 0][1::2] = xs
    diff_table = new_table

    return diff_table

def cubic_spline_matrix(xs, ys):
    n = len(xs) - 1
    matrix = np.zeros((n+1, n+1))
    matrix[0, 0] = 1
    matrix[n, n] = 1

    h_last = xs[1] - xs[0]

    for i in range(1, n):
        h_i = xs[i+1] - xs[i]

        matrix[i, i-1] = h_last
        matrix[i, i] = 2 * (h_i + h_last)
        matrix[i, i+1] = h_i

        h_last = h_i

    return matrix

def cubic_spline_b(xs, ys):
    n = len(xs) - 1
    b = np.zeros(n + 1)

    h_last = xs[1] - xs[0]

    for i in range(1, n):
        h_i = xs[i+1] - xs[i]
        b[i] = 3 * ((ys[i+1] - ys[i]) / h_i - (ys[i] - ys[i-1]) / h_last)
        h_last = h_i

    return b

def cubic_spline_x(xs, ys):
    return np.linalg.solve(cubic_spline_matrix(xs, ys), cubic_spline_b(xs, ys))

