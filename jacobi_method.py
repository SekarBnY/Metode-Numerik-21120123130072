import math

def f(x, y):
    return x**2 + x*y - 10, y + 3*x*(y**2) - 57

def g1(x, y):
    arg = 10 - x * y
    return math.sqrt(arg) if arg >= 0 else float('nan')

def g2(x, y):
    arg = 57 - 3 * x * (y ** 2)
    return math.sqrt(arg) if arg >= 0 else float('nan')

def jacobi(g1, g2, x0, y0, eps=1e-6, maxiter=20, savefile=True):
    x, y = x0, y0
    output = []
    print("\nIterasi Titik Tetap: Metode Jacobi\n")
    output.append("Iterasi Titik Tetap: Metode Jacobi\n")

    for k in range(1, maxiter + 1):
        x_new = g1(x, y)
        y_new = g2(x, y)

        dx = abs(x_new - x)
        dy = abs(y_new - y)

        line = f"Iterasi {k:2d}: x = {x_new:.6f}, y = {y_new:.6f}, deltax = {dx:.6e}, deltay = {dy:.6e}"
        print(line)
        output.append(line)

        if math.isnan(x_new) or math.isnan(y_new):
            warn = "NaN terdeteksi. Iterasi dihentikan (nilai argumen akar negatif)."
            print(warn)
            output.append(warn)
            break

        if abs(x_new) > 1e6 or abs(y_new) > 1e6:
            warn = "Nilai divergen (melebihi batas). Iterasi dihentikan."
            print(warn)
            output.append(warn)
            break

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
    jacobi(g1, g2, x0=1.8, y0=3.2)
