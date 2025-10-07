import numpy as np
import math
import time


def f(x, y):
    return x**2 + x*y - 10, y + 3*x*(y**2) - 57

# Metode Jacobi
def jacobi(g1, g2, x0, y0, eps=1e-6, maxiter=50):
    x, y = x0, y0
    for k in range(1, maxiter + 1):
        x_new = g1(x, y)
        y_new = g2(x, y)
        if math.isnan(x_new) or math.isnan(y_new):
            return None, k, "Divergen (NaN)"
        if abs(x_new - x) < eps and abs(y_new - y) < eps:
            return (x_new, y_new), k, "Konvergen"
        x, y = x_new, y_new
    return None, maxiter, "Tidak konvergen"


def g1(x, y):
    arg = 10 - x*y
    return math.sqrt(arg) if arg >= 0 else float("nan")


def g2(x, y):
    arg = 57 - 3*x*(y**2)
    return math.sqrt(arg) if arg >= 0 else float("nan")


# Metode Seidel
def seidel(g1, g2, x0, y0, eps=1e-6, maxiter=50):
    x, y = x0, y0
    for k in range(1, maxiter + 1):
        x_new = g1(x, y)
        y_new = g2(x_new, y)
        if math.isnan(x_new) or math.isnan(y_new):
            return None, k, "Divergen (NaN)"
        if abs(x_new - x) < eps and abs(y_new - y) < eps:
            return (x_new, y_new), k, "Konvergen"
        x, y = x_new, y_new
    return None, maxiter, "Tidak konvergen"


# Metode Newton-Raphson
def newton_raphson(x0, y0, eps=1e-6, maxiter=50):
    x, y = x0, y0
    for k in range(1, maxiter + 1):
        f1, f2 = f(x, y)
        f1_x, f1_y = 2*x + y, x
        f2_x, f2_y = 3*(y**2), 1 + 6*x*y
        J = np.array([[f1_x, f1_y], [f2_x, f2_y]], dtype=float)
        F = np.array([f1, f2], dtype=float)
        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return None, k, "Jacobian singular"
        x_new = x + delta[0]
        y_new = y + delta[1]
        if abs(delta[0]) < eps and abs(delta[1]) < eps:
            return (x_new, y_new), k, "Konvergen"
        x, y = x_new, y_new
    return None, maxiter, "Tidak konvergen"


# Metode Secant
def secant_2d(p0, p1, eps=1e-6, maxiter=50):
    x0, y0 = p0
    x1, y1 = p1
    for k in range(2, maxiter + 2):
        h = 1e-6
        f1x, f2x = f(x1, y1)
        f1x_hx, f2x_hx = f(x1 + h, y1)
        f1x_hy, f2x_hy = f(x1, y1 + h)
        J = np.array([
            [(f1x_hx - f1x)/h, (f1x_hy - f1x)/h],
            [(f2x_hx - f2x)/h, (f2x_hy - f2x)/h]
        ])
        try:
            delta = np.linalg.solve(J, -np.array([f1x, f2x]))
        except np.linalg.LinAlgError:
            return None, k, "Jacobian tidak stabil"
        x2, y2 = x1 + delta[0], y1 + delta[1]
        if abs(x2 - x1) < eps and abs(y2 - y1) < eps:
            return (x2, y2), k, "Konvergen"
        x0, y0, x1, y1 = x1, y1, x2, y2
    return None, maxiter, "Tidak konvergen"

if __name__ == "__main__":
    print("Perbandingan 4 Metode Penyelesaian Sistem Non-Linear\n")
    x0, y0 = 1.8, 3.2

    # Jacobi
    t0 = time.time()
    result_jacobi, iter_jacobi, status_jacobi = jacobi(g1, g2, x0, y0)
    t1 = time.time()

    # Seidel
    t2 = time.time()
    result_seidel, iter_seidel, status_seidel = seidel(g1, g2, x0, y0)
    t3 = time.time()

    # Newton-Raphson
    t4 = time.time()
    result_newton, iter_newton, status_newton = newton_raphson(x0, y0)
    t5 = time.time()

    # Secant
    t6 = time.time()
    result_secant, iter_secant, status_secant = secant_2d((1.8, 3.2), (1.9, 3.1))
    t7 = time.time()

    def fmt_result(res):
        if res is None:
            return "None"
        else:
            return f"({res[0]:.3f}, {res[1]:.3f})"


    print("=" * 78)
    print(f"{'Metode':<18}{'Hasil (x, y)':<20}{'Iterasi':<10}{'Status':<20}{'Waktu (s)':<10}")
    print("=" * 78)
    print(f"{'Jacobi':<18}{fmt_result(result_jacobi):<20}{iter_jacobi:<10}{status_jacobi:<20}{t1-t0:<10.6f}")
    print(f"{'Seidel':<18}{fmt_result(result_seidel):<20}{iter_seidel:<10}{status_seidel:<20}{t3-t2:<10.6f}")
    print(f"{'Newton-Raphson':<18}{fmt_result(result_newton):<20}{iter_newton:<10}{status_newton:<20}{t5-t4:<10.6f}")
    print(f"{'Secant':<18}{fmt_result(result_secant):<20}{iter_secant:<10}{status_secant:<20}{t7-t6:<10.6f}")
    print("=" * 78)
