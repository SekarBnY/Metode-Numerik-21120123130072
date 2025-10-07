import math
import numpy as np
from tabulate import tabulate

# Persamaan non-linear
def f(x, y):
    """Fungsi sistem non-linear"""
    return x**2 + x*y - 10, y + 3*x*(y**2) - 57

def g1(x, y):
    arg = 10 - x*y
    return math.sqrt(arg) if arg >= 0 else float('nan')

def g2(x, y):
    arg = 57 - 3*x*(y**2)
    return math.sqrt(arg) if arg >= 0 else float('nan')

# Jacobi
def jacobi(g1, g2, x0, y0, eps=1e-6, maxiter=20):
    x, y = x0, y0
    print("\n Iterasi Titik Tetap: Metode Jacobi")
    for k in range(1, maxiter + 1):
        x_new = g1(x, y)
        y_new = g2(x, y)
        print(f"Iter {k:2d}: x = {x_new}, y = {y_new}")

        if math.isnan(x_new) or math.isnan(y_new):
            print("NaN muncul — iterasi tetap dilanjutkan.")
            x_new, y_new = x, y

        if abs(x_new - x) < eps and abs(y_new - y) < eps:
            print(f"\n Konvergen setelah {k} iterasi.")
            print(f"Hasil akhir: x = {x_new:.6f}, y = {y_new:.6f}")
            return (x_new, y_new), k, "Konvergen"

        x, y = x_new, y_new

    print("\n Tidak konvergen (semua iterasi ditampilkan).")
    return (x, y), maxiter, "Tidak"


#  Seidel

def seidel(g1, g2, x0, y0, eps=1e-6, maxiter=20):
    x, y = x0, y0
    print("\n Iterasi Titik Tetap: Metode Seidel")
    for k in range(1, maxiter + 1):
        x_new = g1(x, y)
        y_new = g2(x_new, y)
        print(f"Iter {k:2d}: x = {x_new}, y = {y_new}")

        if math.isnan(x_new) or math.isnan(y_new):
            print(" NaN muncul — iterasi tetap dilanjutkan.")
            x_new, y_new = x, y

        if abs(x_new - x) < eps and abs(y_new - y) < eps:
            print(f"\n Konvergen setelah {k} iterasi.")
            print(f"Hasil akhir: x = {x_new:.6f}, y = {y_new:.6f}")
            return (x_new, y_new), k, "Konvergen"

        x, y = x_new, y_new

    print("\n Tidak konvergen (semua iterasi ditampilkan).")
    return (x, y), maxiter, "Tidak"

# Newton-Raphson

def newton_raphson(x0, y0, eps=1e-6, maxiter=20):
    x, y = x0, y0
    print("\n Metode Newton-Raphson ")

    for k in range(1, maxiter + 1):
        u, v = f(x, y)
        u_x, u_y = 2*x + y, x
        v_x, v_y = 3*(y**2), 1 + 6*x*y
        J = np.array([[u_x, u_y], [v_x, v_y]])
        F = np.array([u, v])

        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            print(f" Jacobian singular pada iterasi {k}")
            continue

        x_new, y_new = x + delta[0], y + delta[1]
        print(f"Iter {k:2d}: x = {x_new}, y = {y_new}")

        if abs(x_new - x) < eps and abs(y_new - y) < eps:
            print(f"\n Konvergen setelah {k} iterasi.")
            print(f"Hasil akhir: x = {x_new:.6f}, y = {y_new:.6f}")
            return (x_new, y_new), k, "Konvergen"

        x, y = x_new, y_new

    print("\n Tidak konvergen (semua iterasi ditampilkan).")
    return (x, y), maxiter, "Tidak"

# Secant-like

def secant_2d(p0, p1, eps=1e-6, maxiter=20):
    x0, y0 = p0
    x1, y1 = p1
    print("\n Metode Secant")

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
            print(f" Jacobian tidak stabil pada iterasi {k}")
            continue

        x2, y2 = x1 + delta[0], y1 + delta[1]
        print(f"Iter {k:2d}: x = {x2}, y = {y2}")

        if abs(x2 - x1) < eps and abs(y2 - y1) < eps:
            print(f"\n Konvergen setelah {k} iterasi.")
            print(f"Hasil akhir: x = {x2:.6f}, y = {y2:.6f}")
            return (x2, y2), k, "Konvergen"

        x0, y0, x1, y1 = x1, y1, x2, y2

    print("\n Tidak konvergen (semua iterasi ditampilkan).")
    return (x2, y2), maxiter, "Tidak"


def main():
    x0, y0 = 1.8, 3.2
    print("     PERBANDINGAN METODE NON-LINEAR")
    print(f"Tebakan awal digunakan: x0={x0}, y0={y0}\n")

    results = []
    results.append(["Jacobi", *jacobi(g1, g2, x0, y0)[1:]])
    results.append(["Seidel", *seidel(g1, g2, x0, y0)[1:]])
    results.append(["Newton-Raphson", *newton_raphson(x0, y0)[1:]])
    results.append(["Secant", *secant_2d((x0, y0), (x0+0.1, y0-0.1))[1:]])

    print("\n RINGKASAN HASIL")
    print(tabulate(results, headers=["Metode", "Iterasi", "Status"], tablefmt="grid"))

if __name__ == "__main__":
    main()
