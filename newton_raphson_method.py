import numpy as np
import math

def f(x, y):
    return x**2 + x*y - 10, y + 3*x*(y**2) - 57

def newton_raphson(x0, y0, eps=1e-6, maxiter=20, savefile=True):
    print("\n Metode Newton-Raphson ")
    output = []
    x, y = x0, y0
    for k in range(1, maxiter + 1):
        u, v = f(x, y)
        u_x, u_y = 2*x + y, x
        v_x, v_y = 3*(y**2), 1 + 6*x*y
        J = np.array([[u_x, u_y], [v_x, v_y]])
        F = np.array([u, v])

        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            warn = f" Jacobian singular pada iterasi {k}"
            print(warn)
            output.append(warn)
            continue

        x_new, y_new = x + delta[0], y + delta[1]
        line = f"Iter {k:2d}: x = {x_new}, y = {y_new}"
        print(line)
        output.append(line)

        if abs(x_new - x) < eps and abs(y_new - y) < eps:
            result = f"\n Konvergen setelah {k} iterasi.\nHasil akhir: x ≈ {x_new:.6f}, y ≈ {y_new:.6f}"
            print(result)
            output.append(result)
            break

        x, y = x_new, y_new

    else:
        result = "\n Tidak konvergen (semua iterasi ditampilkan)."
        print(result)
        output.append(result)

if __name__ == "__main__":
    newton_raphson(x0=1.8, y0=3.2)
