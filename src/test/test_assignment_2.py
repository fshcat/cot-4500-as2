from src.main.assignment_2 import *

if __name__ == "__main__":
    # Neville test
    xs = [3.6, 3.8, 3.9]
    ys = [1.675, 1.436, 1.318]
    print(neville_method(xs, ys, 3.7))
    print()

    # Newton forward difference coefficients
    xs = [7.2, 7.4, 7.5, 7.6]
    ys = [23.5492, 25.3913, 26.8224, 27.4589]
    
    table = newton_forward_difference(xs, ys)
     
    for i in range(1,len(table)):
        print(table[i, i])
    print()

    # Evaluating above coefficients at x=7.3
    print(newton_forward_interpolation(xs, ys, 7.3))
    print()

    # Hermite forward difference matrix test
    xs = [3.6, 3.8, 3.9]
    ys = [1.675, 1.436, 1.318]
    dys = [-1.195, -1.188, -1.182]

    table = hermite_forward_difference(xs, ys, dys)
        
    for row in table:
        formatted_row = []
        for x in row:
            if x >= 0:
                formatted_row.append(f" {x:14.8e}")
            else:
                formatted_row.append(f"{x:14.8e}")  

        print(f"[{' '.join(formatted_row)}]")
    print()

    # Cubic spline interpolation test
    xs = [2, 5, 8, 10]
    ys = [3, 5, 7, 9]

    A = cubic_spline_matrix(xs, ys)
    b = cubic_spline_b(xs, ys)
    x = cubic_spline_x(xs, ys)

    print(A)
    print(b)
    print(x)
