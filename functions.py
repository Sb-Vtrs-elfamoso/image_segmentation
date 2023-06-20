import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.integrate import simpson
from mpl_toolkits.mplot3d import Axes3D


def fast_marching(mask):
    nbc, nbl = mask.shape[:2]
    D = np.ones((nbc, nbl)) * 1000000.0

    for i in range(nbc):
        for j in range(nbl):
            if mask[i, j] == 1:
                D[i, j] = 0

    for j in range(nbl):
        for i in range(nbc):
            if i > 0:
                if D[i - 1, j] + 1.0 < D[i, j]:
                    D[i, j] = D[i - 1, j] + 1.0
            if j > 0:
                if D[i, j - 1] + 1.0 < D[i, j]:
                    D[i, j] = D[i, j - 1] + 1.0
            if i > 0 and j > 0:
                if D[i - 1, j - 1] + np.sqrt(2.0) < D[i, j]:
                    D[i, j] = D[i - 1, j - 1] + np.sqrt(2.0)
            if i < nbc - 1 and j > 0:
                if D[i + 1, j - 1] + np.sqrt(2.0) < D[i, j]:
                    D[i, j] = D[i + 1, j - 1] + np.sqrt(2.0)

    for i in range(nbc - 2, -1, -1):
        for j in range(nbl):
            if D[i + 1, j] + 1.0 < D[i, j]:
                D[i, j] = D[i + 1, j] + 1.0

    for j in range(nbl - 1, -1, -1):
        for i in range(nbc - 1, -1, -1):
            if i < nbc - 1:
                if D[i + 1, j] + 1.0 < D[i, j]:
                    D[i, j] = D[i + 1, j] + 1.0
            if j < nbl - 1:
                if D[i, j + 1] + 1.0 < D[i, j]:
                    D[i, j] = D[i, j + 1] + 1.0
            if i < nbc - 1 and j < nbl - 1:
                if D[i + 1, j + 1] + np.sqrt(2.0) < D[i, j]:
                    D[i, j] = D[i + 1, j + 1] + np.sqrt(2.0)
            if i > 0 and j < nbl - 1:
                if D[i - 1, j + 1] + np.sqrt(2.0) < D[i, j]:
                    D[i, j] = D[i - 1, j + 1] + np.sqrt(2.0)
    for i in range(1, nbc):
        if D[i - 1, j] + 1.0 < D[i, j]:
            D[i, j] = D[i - 1, j] + 1.0

    return D



def signed_distance_from_mask(mask):
    Dm = fast_marching(mask)
    Dp = fast_marching(1 - mask)
    phi = Dp - Dm
    return phi


def gradx(I):
    m, n = I.shape
    M = np.zeros((m, n))
    M[:-1, :] = I[1:, :] - I[:-1, :]
    return M


def grady(I):
    m, n = I.shape
    M = np.zeros((m, n))
    M[:, :-1] = I[:, 1:] - I[:, :-1]
    return M

def div(px, py):
    m, n = px.shape
    M = np.zeros((m, n))
    Mx = np.zeros((m, n))
    My = np.zeros((m, n))

    Mx[1:m-1, :] = px[1:m-1, :] - px[:m-2, :]
    Mx[0, :] = px[0, :]
    Mx[m-1, :] = -px[m-2, :]

    My[:, 1:n-1] = py[:, 1:n-1] - py[:, :n-2]
    My[:, 0] = py[:, 0]
    My[:, n-1] = -py[:, n-2]

    M = Mx + My
    return M

def delta_eta(phi, eta):
    M = 1. / ((1 + (phi / eta) ** 2) * eta * np.pi)
    return M


def heavyside_eta(phi, eta):
    M = (1. + 2. * np.arctan(phi / eta) / np.pi) / 2.
    return M

#Function to compute graf_phi
def grad_phi(img, phi, eta, eps, lamb, c1, c2):
    gx, gy = gradx(phi), grady(phi)
    norm = np.sqrt(np.sum(gx**2 + gy**2 + eps**2))
    px, py = gx/norm, gy/norm
    return -delta_eta(phi, eta) * (div(px, py) + lamb*((img - c2)**2 - (img - c1)**2))


# Function to update color constants c1 and c2
def update_color_constants(phi, image,eta):
    #compute Heavyside
    H = heavyside_eta(phi, eta)
    #compute c1
    numerator1 = np.sum(H * image)
    denominator1 = np.sum(H)
    c1=numerator1 / denominator1
    #compute c2
    numerator2 = np.sum((1-H) * image)
    denominator2 = np.sum(1-H)
    c2=numerator2 / denominator2
    
    return c1, c2

def update_phi(img, phi, eta=1, eps=1, lamb=1e-4, n=1500, threshold=4e-1):    
    #update the color constant c1 and c2
    c1,c2 = update_color_constants(phi, img,eta)
    #calculate de gradient 
    gradient = grad_phi(img, phi, eta, eps, lamb, c1, c2)
    #calculate the step
    tau = 1/(2*np.max(np.max(gradient)))
    #desente de gradient method 
    new_phi = phi - tau*gradient
    phi = new_phi
    return phi

def Chan_Vese_level_set_formulation(image, phi, eta, epsilon, lambda_val, n_iterations, threshold=4e-1) :
    phi_sequence = [phi.copy()]

    # Level set evolution
    for i in range(n_iterations):
    

    
        new_phi = update_phi(image, phi, eta, epsilon, lambda_val, n_iterations, threshold=4e-1)

        if i % 10 == 0:
            new_phi = signed_distance_from_mask(new_phi>0)
            phi_sequence.append(phi.copy())
    
        dif = np.linalg.norm(heavyside_eta(new_phi, eta) - heavyside_eta(phi, eta))
        phi= new_phi
    
    
        if  i>100 and dif < threshold:
            print(f'dif={dif} and i={i}')
            break
    return phi,phi_sequence
    
def extract_elements(phi_sequence):
    n = len(phi_sequence)
    if n <= 10:
        return phi_sequence
    else:
        step = (n - 1) // 8
        indices = np.arange(0, n, step)
        indices = np.unique(np.concatenate(([0], indices, [n - 1])))
        extracted_elements = [phi_sequence[i] for i in indices]
        return extracted_elements
        
        
        
def int_2D(zz):
    
    dim = zz.shape
    Y = np.arange(dim[0])
    X = np.arange(dim[1])
    return simpson([simpson(zz_x,X) for zz_x in zz],Y)
    
    
def projection(u) : 
    return np.minimum(np.maximum(u, 0), 1)
    
def compute_energy_smooth(mask,foreground,background,lamb,eps=0.000001):
    gradient_u_x, gradient_u_y = gradx(mask), grady(mask)
    norm_gradient_u_eps = np.sqrt((gradient_u_x**2 + gradient_u_y**2 + eps**2))
    
   
    return int_2D(norm_gradient_u_eps*np.ones(mask.shape)) + lamb * int_2D(np.abs(foreground) * mask)+lamb * int_2D(np.abs(background) * (1 - mask))
    
    
def projected_gradient(mask, grey_img, tau, lamb, c1, c2, nb_iter, threshold, eps=0.000001) : 
    mask_copy = np.copy(mask)
    foreground = (grey_img-c1)**2
    background = (grey_img-c2)**2
    functional = []
    for i in range(nb_iter) : 
        gradient_u_x, gradient_u_y = gradx(mask_copy), grady(mask_copy)
        norm_gradient = np.sqrt(np.sum(gradient_u_x**2 + gradient_u_y**2 + eps**2))
        px, py = gradient_u_x/norm_gradient, gradient_u_y/norm_gradient
        d = div(px,py)
        res = mask_copy + tau*(d-lamb*foreground + lamb*background)
        
        mask_copy = projection(res)
        
        energy = compute_energy_smooth(mask_copy,foreground,background,lamb,eps=0.000000001)
        
        functional.append(energy)
    
    binary_mask = np.where(mask_copy >= threshold, 1, 0)
    print(lamb)
    plt.imshow(binary_mask, cmap='hot', interpolation='nearest')
    plt.show()
    return binary_mask, functional
    
def detect_contour(M):
    # Detect the frontier M=0
    dim = M.shape
    L = []
    for j in range(dim[1]):
        for i in range(dim[0]):
            b = True
            if not M[i,j]:
                continue
            if i > 0:
                if j > 0:
                    b = b and M[i-1, j-1]
                b = b and M[i-1, j]
                if j < dim[1] -1:
                    b = b and M[i-1, j+1]
            if j > 0:
                b = b and M[i, j-1]
            if j < dim[1] -1:
                b = b and M[i, j+1]
            if i < dim[0] - 1:
                if j > 0:
                    b = b and M[i+1, j-1]
                b = b and M[i+1, j]
                if j < dim[1] -1:
                    b = b and M[i+1, j+1]
            if not b:
                L.append(np.array([i,j]))
    return np.array(L)
    
    
    
def projB(z) :
    norm = np.linalg.norm(z)
    if norm <= 1 : r=z
    else : r=z/norm
    return r 

def projA(u) : 
    return(np.minimum(np.maximum(u, 0), 1))

def update_z(u, z, sigma, c1, c2) :

    gradxy_u = np.array([gradx(u), grady(u)])
    z_new = np.zeros_like(gradxy_u)  # taille (2, 242, 308)
    for j in range(gradxy_u.shape[1]) :
        for k in range(gradxy_u.shape[2]) : 
            z_new[:,j,k] = projB(z[:,j,k] + sigma*gradxy_u[:,j,k])
    # la projection se fait terme à terme : ( z dans R^2)
    # pour chaque pixel (position dans l'image) applique projB au couple de valeurs des deux images (gradx, grady)
    return z_new
    
    
    
def test_dual(u0,z0, image, tau, sigma, lamb, c1, c2, nb_iter, eps, threshold) : 
    u = np.copy(u0)
    z = np.copy(z0)
    # func = np.zeros((nb_iter, 3))
    #func = []
    #f1, f2, f3 = (energy_dual(u, image, c1, c2, lamb, eps))
    n=0
    #func.append([f1, f2, f3])
    dif = 10000
    while n < nb_iter and dif > threshold : 
        n+=1
        c1, c2 = update_color_constants(u, image,eta=0.001)
        z_new = update_z(u, z, sigma, c1, c2)
        grad_u = -div(z_new[0,:,:], z_new[1,:,:]) + lamb*((image - c1)**2 - (image - c2)**2)
        u = projA(u - tau*grad_u)
        z = z_new
        #f1, f2, f3 = energy_dual(u, image, c1, c2, lamb, eps)
        #func.append([f1, f2, f3])
        #dif = abs((func[-1][0] + func[-1][1] + func[-1][2]) - (func[-2][0]+func[-2][1]+func[-2][2]))
        # print(dif)
        if (n% 200) == 0 : 
            plt.imshow(u)
            plt.show()
        #print(func.shape)
    return u#, func
