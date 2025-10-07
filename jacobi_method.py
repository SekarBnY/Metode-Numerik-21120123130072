import math

def f(x, y):
    return x**2 + x*y - 10, y + 3*x*(y**2) - 57

def g1(x, y):
    arg = 10 - x*y
    return math.sqrt(arg) if arg >= 0 else float('nan')

def g2(x, y):
    arg = 57 - 3*x*(y**2)
    return math.sqrt(arg) if arg >= 0 else float('nan')

def jacobi(g1, g2, x0, y0, eps=1e-6, maxiter=20, savefile=True):
    x, y = x0, y0
    output = []
    print("\n Iterasi Titik Tetap: Metode Jacobi")
    for k in range(1, maxiter + 1):
        x_new = g1(x, y)
        y_new = g2(x, y)
        line = f"Iter {k:2d}: x = {x_new}, y = {y_new}"
        print(line)
        output.append(line)

        if math.isnan(x_new) or math.isnan(y_new):
            warn = " NaN (Not a Number) muncul — iterasi tetap dilanjutkan untuk observasi."
            print(warn)
            output.append(warn)
            x_new, y_new = x, y

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
    jacobi(g1, g2, x0=1.8, y0=3.2)
