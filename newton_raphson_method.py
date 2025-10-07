import numpy as np
import math

def f(x, y):
    return x**2 + x*y - 10, y + 3*x*(y**2) - 57


def newton_raphson(x0, y0, eps=1e-6, maxiter=20, savefile=True):
    print("\nMetode Newton-Raphson\n")
    output = ["Metode Newton-Raphson\n"]

    x, y = x0, y0

    for k in range(1, maxiter + 1):

        f1, f2 = f(x, y)


        f1_x = 2*x + y
        f1_y = x
        f2_x = 3*(y**2)
        f2_y = 1 + 6*x*y


        J = np.array([[f1_x, f1_y],
                      [f2_x, f2_y]], dtype=float)
        F = np.array([f1, f2], dtype=float)

        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            warn = f"Jacobian singular pada iterasi {k}. Iterasi dihentikan."
            print(warn)
            output.append(warn)
            break

        x_new = x + delta[0]
        y_new = y + delta[1]

        dx = abs(delta[0])
        dy = abs(delta[1])

        line = f"Iterasi {k:2d}: x = {x_new:.6f}, y = {y_new:.6f}, dx = {dx:.6e}, dy = {dy:.6e}"
        print(line)
        output.append(line)

  
        if dx < eps and dy < eps:
            result = f"\nKonvergen setelah {k} iterasi.\nHasil akhir: x = {x_new:.6f}, y = {y_new:.6f}"
            print(result)
            output.append(result)
            break

        x, y = x_new, y_new

    else:
        result = "\nTidak konvergen setelah iterasi maksimum."
        print(result)
        output.append(result)

if __name__ == "__main__":
    newton_raphson(x0=1.8, y0=3.2)
