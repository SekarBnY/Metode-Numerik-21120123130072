import numpy as np
import math


def f(x, y):
    return x**2 + x*y - 10, y + 3*x*(y**2) - 57

def secant_2d(p0, p1, eps=1e-6, maxiter=20, savefile=True):
    print("\nMetode Secant\n")
    output = ["Metode Secant\n"]

    x0, y0 = p0
    x1, y1 = p1

    for k in range(2, maxiter + 2):
        h = 1e-6 
        f1x, f2x = f(x1, y1)


        f1x_hx, f2x_hx = f(x1 + h, y1)
        f1x_hy, f2x_hy = f(x1, y1 + h)

        J = np.array([
            [(f1x_hx - f1x) / h, (f1x_hy - f1x) / h],
            [(f2x_hx - f2x) / h, (f2x_hy - f2x) / h]
        ])

        try:
            delta = np.linalg.solve(J, -np.array([f1x, f2x]))
        except np.linalg.LinAlgError:
            warn = f"Jacobian tidak stabil pada iterasi {k}. Iterasi dihentikan."
            print(warn)
            output.append(warn)
            break

        x2 = x1 + delta[0]
        y2 = y1 + delta[1]

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        line = f"Iterasi {k:2d}: x = {x2:.6f}, y = {y2:.6f}, dx = {dx:.6e}, dy = {dy:.6e}"
        print(line)
        output.append(line)

        if dx < eps and dy < eps:
            result = f"\nKonvergen setelah {k} iterasi.\nHasil akhir: x = {x2:.6f}, y = {y2:.6f}"
            print(result)
            output.append(result)
            break

        x0, y0, x1, y1 = x1, y1, x2, y2

    else:
        result = "\nTidak konvergen setelah iterasi maksimum."
        print(result)
        output.append(result)


if __name__ == "__main__":
    secant_2d((1.8, 3.2), (1.9, 3.1))
