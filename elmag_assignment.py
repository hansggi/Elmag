import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import time

L = 1

N = 300

def fourier_coeff(V_0, num_terms):
    c_arr = np.zeros(num_terms)
    for i in range(1, num_terms + 1):
        c_arr[i-1] = one_coeff(V_0, i)
    return c_arr 
def one_coeff(V_0, term_num):
    k = term_num * np.pi
    c = 2  / np.sinh(k) * integrate.quad(lambda x: np.sin(k * x ) * V_0(x), 0, 1)[0] 
    return c

def get_V(x, y, c_coeff):
    V_arr = np.zeros((len(y), len(x)))
    xx, yy = np.meshgrid(x, y)
    for i in range(1, len(c_coeff) + 1):
        k = i * np.pi
        V_arr +=np.sin(k * xx) * c_coeff[i-1] * np.sinh(k * yy)
    return xx, yy, V_arr
def one_V(xx, yy, i, c):
    k = i * np.pi
    return np.sin(k * xx) * c * np.sinh(k * yy)

def plot_pot(N, num_terms, V_0, get_V, plot, fn):
    c_arr = fourier_coeff(V_0, num_terms)
    x = np.linspace(0, L ,N)
    y = np.linspace(0, L, N)
    xx, yy, V_arr = get_V(x, y,  c_arr)
    if plot:
        fig = plt.figure(figsize= plt.figaspect(0.33))
        ax1 = fig.add_subplot(131,  projection='3d')
        ax1.plot_surface(xx, yy, V_arr,cmap = "viridis")
        ax1.set_xlabel("$\\xi_x$")
        ax1.set_ylabel("$\\xi_y$")
        ax1.set_zlabel("$V/V_c$", ) 

        ax2 = fig.add_subplot(132)
        ax2.plot(x, V_0(x), color = "blue", label = "Boundary condition at y = L")
        ax2.plot(x, V_arr[N-1, :], color = "red", label = "Potential at y = L")
        ax2.legend(fontsize = "large", loc = "best")
        ax2.set_xlabel("$\\xi_x$")
        ax2.set_ylabel("$V/V_c$")
        
        ax3 = fig.add_subplot(133)
        grad = np.gradient(V_arr)

        num_arrows = 20 #in each dimension
        a = int(N / num_arrows)
        magnitude = np.sqrt(grad[0] ** 2 + grad[1]**2) 
        CS = ax3.contourf(xx, yy, magnitude**(1/2))
        
        
        cbar = fig.colorbar(CS)
        cbar.ax.set_ylabel("$|E|/V_c$")
        
        magnitude[0,0] = 1e-12 #to avoid dividing by zero
        ax3.quiver(xx[::a, ::a], yy[::a, ::a], - grad[1][::a, ::a] / magnitude[::a, ::a], - grad[0][::a, ::a] / magnitude[::a, ::a])
        ax3.set_xlabel("$\\xi_x$")
        ax3.set_ylabel("$\\xi_y$")
        plt.tight_layout()
        plt.savefig(fn)
        plt.show()

def V_1(x):
    return np.sin(10 *np.pi * x / L)
def V_2(x):
    return (1 - (x - 1/2)**4) 

def V_3(x):
    return np.heaviside(x - 0.5, 0.5) * np.heaviside(3 * L / 4 - x, 0.5) #Second argument is the value of the function when the first argument is zero. Standard to set to 0.5
def convergence_total(N, V_1, V_2, V_3, get_V, num_exact, plot): 
    x = np.linspace(0, L , N)
    y = np.linspace(0, L, N)
    err_arr = np.zeros((3, num_exact))
    cou = -1
    for V in (V_1, V_2, V_3):
        
        c_arr_exact = fourier_coeff(V, num_exact)
        xx, yy, V_arr_ex = get_V(x, y, c_arr_exact)
        err_ex = np.linalg.norm(V_arr_ex)

        cou +=1
        err_arr[cou, 0 ] = 1 #err_ex / err_ex
        V_tot = np.zeros(V_arr_ex.shape)
        for i in range(1, num_exact):
            c = one_coeff(V, term_num = i)
            dV = one_V(xx, yy, i,c)
            V_tot += dV
            err = (V_tot - V_arr_ex)
            err_arr[cou, i] = np.linalg.norm(err) / err_ex

    if plot:
        plt.title("Convergence") 
        plt.xlabel("Number of fourier terms")
        plt.ylabel("Relative error")
        plt.plot(np.arange(1, num_exact+1, 1), err_arr[0], label = "$\sin(10 \pi \\xi_x)$")
        plt.plot(np.arange(1, num_exact+1, 1), err_arr[1], label = "$(1 - (\\xi_x - 1/2)^4)$")
        plt.plot(np.arange(1, num_exact+1, 1), err_arr[2], label = "$\\theta(\\xi_x -1/2) \\theta(3/4 - \\xi_x)$")
        plt.legend()
        #plt.yticks(np.arange(0, 1.1, 0.1))
        plt.savefig("conv_abs.pdf")
        plt.show() 

def convergence_diff(N, V_1, V_2, V_3, get_V, num_terms, plot):
    x = np.linspace(0, L , N)
    y = np.linspace(0, L, N)
    err_arr = np.zeros((3, num_terms ))
    cou = -1
    for V in (V_1, V_2, V_3):
        c_norm = fourier_coeff(V, num_terms)
        V_normalize = get_V(x, y, c_norm)[2]
        err_norm = np.linalg.norm(V_normalize)
        
        cou +=1
        c_arr_old = fourier_coeff(V, 0) 
        xx, yy, V_old = get_V(x, y, c_arr_old)
        norm_tot = 0
        for i in range(0, num_terms):
            c = one_coeff(V, i+1)
            dV = one_V(xx, yy, i+1, c ) 
            V_new = V_old + dV
            norm_tot +=dV
            err = (V_new - V_old)
            err = (V_new - V_old)
            err_arr[cou, i] = np.linalg.norm(err) / err_norm
            V_old = V_new
    if plot:
        plt.title("Convergence difference") 
        plt.xlabel("Number of fourier terms")
        plt.ylabel("Error from last step")
        plt.ylim(0, 0.1)
        plt.yticks(np.arange(0, 0.1, 0.01))
        plt.plot(np.arange(1, num_terms+1, 1), err_arr[0], label = "$\sin(10 \pi \\xi_x)$" )
        plt.plot(np.arange(1, num_terms+1, 1), err_arr[1], label = "$(1 - (\\xi_x - 1/2)^4)$")
        plt.plot(np.arange(1, num_terms+1, 1), err_arr[2], label = "$\\theta(\\xi_x - 1/2) \\theta(3/4 - \\xi_x )$")
        plt.plot(np.arange(1, num_terms+1, 1), np.ones(num_terms) * 0.01)
        plt.legend()
        plt.savefig("conv_diff_zoom.pdf")

        plt.show() 

plot_pot(N, 100, V_1, get_V, True, "V_1.pdf")
plot_pot(N, 100, V_2, get_V, True, "V_2.pdf")
plot_pot(N, 100, V_3, get_V, True, "V_3.pdf")

convergence_diff(N,V_1,V_2, V_3, get_V, 100, True)
convergence_total(N, V_1,V_2, V_3, get_V, 100, True)