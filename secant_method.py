import numpy as np
import math

def f(x, y):
    return x**2 + x*y - 10, y + 3*x*(y**2) - 57

def secant_2d(p0, p1, eps=1e-6, maxiter=20, savefile=True):
    print("\n  Metode Secant ")
    output = []
    x0, y0 = p0
    x1, y1 = p1

    for k in range(2, maxiter + 2):
        h = 1e-6
        ux, vx = f(x1, y1)
        ux_hx, vx_hx = f(x1 + h, y1)
        ux_hy, vx_hy = f(x1, y1 + h)
        J = np.array([
            [(ux_hx - ux)/h, (ux_hy - ux)/h],
            [(vx_hx - vx)/h, (vx_hy - vx)/h],
        ])
        try:
            delta = np.linalg.solve(J, -np.array([ux, vx]))
        except Exception:
            warn = f"Jacobian tidak stabil pada iterasi {k}"
            print(warn)
            output.append(warn)
            continue

        x2, y2 = x1 + delta[0], y1 + delta[1]
        line = f"Iter {k:2d}: x = {x2}, y = {y2}"
        print(line)
        output.append(line)

        if abs(x2 - x1) < eps and abs(y2 - y1) < eps:
            result = f"\n Konvergen setelah {k} iterasi.\nHasil akhir: x ≈ {x2:.6f}, y ≈ {y2:.6f}"
            print(result)
            output.append(result)
            break

        x0, y0, x1, y1 = x1, y1, x2, y2

    else:
        result = "\n Tidak konvergen (semua iterasi ditampilkan)."
        print(result)
        output.append(result)

if __name__ == "__main__":
    secant_2d((1.8, 3.2), (1.9, 3.1))
