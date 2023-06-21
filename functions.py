import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson


#------------------------------------------------------------------------------
#                               Fonctions Utiles
#------------------------------------------------------------------------------

# Affichage masque et image
def affichage(ax, image, mask, title):
    ax.imshow(image, cmap='gray')
    ax.contour(mask, colors='r')
    ax.set_title(title)

# Gradient selon x
def gradx(I):
    m, n = I.shape
    M = np.zeros((m, n))
    M[:-1, :] = I[1:, :] - I[:-1, :]
    return M

# Gradient selon y
def grady(I):
    m, n = I.shape
    M = np.zeros((m, n))
    M[:, :-1] = I[:, 1:] - I[:, :-1]
    return M

# Divergence
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

# Calcul d'integrale
def int_2D(foo):
    dim = foo.shape
    Y = np.arange(dim[0])
    X = np.arange(dim[1])
    return simpson([simpson(foo_x,X) for foo_x in foo],Y)
    
# Projection sur l'espace du masque {0,1}^Omega
def projection(u) : 
    return np.minimum(np.maximum(u, 0), 1)

# Calcul de l'energie (aka fonctionnelle)
def energie(Du,mask,foreground,background,lamb,eps=0.000001):
    return int_2D(Du) + lamb * int_2D(np.abs(foreground) * mask)+lamb * int_2D(np.abs(background) * (1 - mask))

#------------------------------------------------------------------------------
#                         Chan-Vese level set formulation
#------------------------------------------------------------------------------

# Calcul des valeur de phi à partir d'un masque (à l'intérieur ou à l'extérieur)
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

# Calcul de phi à partir d'un masque
def signed_distance_from_mask(mask):
    Dm = fast_marching(mask)
    Dp = fast_marching(1 - mask)
    phi = Dp - Dm
    return phi

# Conversion de phi en masque
def phi_to_mask(phi):
    mask = np.zeros_like(phi)
    mask[phi > 0] = 1
    return mask


# Visualisation de phi
def visualize_phi(ax, phi, title):
    x, y = np.meshgrid(range(phi.shape[1]), range(phi.shape[0]))
    ax.plot_surface(x, y, phi, cmap='jet', rstride=1, cstride=1, alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Phi')
    
    # Contour pour phi=0
    ax.contour(x, y, phi, levels=[0], colors='red', linewidths=5, alpha=1)
    ax.set_title(title)

# Gradient de phi
def grad_phi(img, phi, eta, eps):
    gx, gy = gradx(phi), grady(phi)
    norm = np.sqrt(gx**2 + gy**2 + eps**2)
    px, py = gx/norm, gy/norm
    return -delta_eta(phi, eta), (div(px, py)), norm


# Mise à jour des valeurs des couleurs c1(1er plan) et c2(arriere plan)
def update_color_constants(phi, image, eta):

    H = heavyside_eta(phi, eta)

    c1 = np.sum(H * image) / np.sum(H)

    c2 = np.sum((1-H) * image) / np.sum(1-H)
    
    return c1, c2

# Mise à jour de phi par descente de gradient
def update_phi(img, phi, eta, eps, lamb):    
    # Maj c1 et c2
    c1,c2 = update_color_constants(phi, img, eta)
    
    foreground = (img-c1)**2
    background = (img-c2)**2
    
    # descente de gradient
    delta_H, delta_phi, norm_delta_phi = grad_phi(img, phi, eta, eps)
    delta_J = delta_H*(delta_phi - lamb*foreground + lamb*background)
    tau = 1/(2*np.max(np.max(delta_J)))
    new_phi = phi - tau*delta_J
    
    return new_phi, -delta_H*norm_delta_phi, foreground, background

def update_phi_fista(img, phi, old_phi,i, eta=1, eps=1, lamb=1e-4, n=10):
    # Maj c1 et c2
    c1,c2 = update_color_constants(phi, img,eta)
    
    foreground = (img-c1)**2
    background = (img-c2)**2
    
    # descente de gradient acceleree
    alpha=(lambda n : n/(n+3))
    y = phi + alpha(i)*(phi - old_phi)

    delta_H, delta_phi, norm_delta_phi = grad_phi(img, y, eta, eps)
    delta_J = delta_H*(delta_phi - lamb*foreground + lamb*background)
    tau = 1/(2*np.max(np.max(delta_J)))
    new_phi = y - tau*delta_J
    phi = new_phi
    
    return y, phi, -delta_H*norm_delta_phi, foreground, background

def Chan_Vese_level_set_formulation(image, phi, eta, epsilon, lambda_val, n_iterations, threshold, verbose=False) :
    phi_sequence = [phi.copy()]
    functional = []

    # Level set evolution
    for i in range(n_iterations):
    
        new_phi, norm_gradient, foreground, background = update_phi(image, phi, eta, epsilon, lambda_val)
        energy = energie(norm_gradient,new_phi>0,foreground,background,lambda_val,epsilon)
        functional.append(energy)
        
        if i % 10 == 0:
            new_phi = signed_distance_from_mask(new_phi>0)
            phi_sequence.append(phi.copy())
    
        dif = np.linalg.norm(heavyside_eta(new_phi, eta) - heavyside_eta(phi, eta))
        phi = new_phi
        if verbose :
            print(f'dif={dif} and i={i}')

    
        if  i>100 and dif < threshold:
            print(f'dif={dif} and i={i}')
            break
    return phi, phi_sequence, functional
    
def Chan_Vese_level_set_fista_formulation(image, phi, eta, epsilon, lambda_val, n_iterations, threshold, verbose=False) :
    phi_sequence = [phi.copy()]
    old_phi = phi
    functional = []
    
    for i in range(n_iterations):
    
        old_phi, new_phi, norm_gradient, foreground, background = update_phi_fista(image, phi, old_phi, i, eta, epsilon, lambda_val, n_iterations)
        energy = energie(norm_gradient,new_phi>0,foreground,background,lambda_val,epsilon)
        functional.append(energy)
        
        if i % 10 == 0:
            new_phi = signed_distance_from_mask(new_phi>0)
            phi_sequence.append(phi.copy())
    
        dif = np.linalg.norm(heavyside_eta(new_phi, eta) - heavyside_eta(phi, eta))
        phi = new_phi
        if verbose :
            print(f'dif={dif} and i={i}')

    
        if  i>100 and dif < threshold:
            print(f'dif={dif} and i={i}')
            break
    return phi, phi_sequence, i, functional

def extract_elements(phi_sequence):
    step = int(len(phi_sequence) / 10)
    selected_elements = []
    index = 0
    indices=[]

    while len(selected_elements) < 10 and index<len(phi_sequence):
        selected_elements.append(phi_sequence[index])
        indices.append(index*10)
        index += step
        

    return selected_elements, indices
        
#------------------------------------------------------------------------------
#                  Chan, Esedoglu and Nikolova convex formulation
#------------------------------------------------------------------------------
    
    
def projected_gradient(mask, img, tau, lamb, c1, c2, nb_iter, threshold, eps=0.000001) : 
    mask_copy = np.copy(mask)
    foreground = (img-c1)**2
    background = (img-c2)**2
    functional = []
    for i in range(nb_iter) : 
        gradient_u_x, gradient_u_y = gradx(mask_copy), grady(mask_copy)
        norm_gradient = np.sqrt(gradient_u_x**2 + gradient_u_y**2 + eps**2)
        px, py = gradient_u_x/norm_gradient, gradient_u_y/norm_gradient
        d = div(px,py)
        res = mask_copy + tau*(d-lamb*foreground + lamb*background)
        
        mask_copy = projection(res)
        
        energy = energie(norm_gradient,mask_copy,foreground,background,lamb,eps=eps)
        
        functional.append(energy)
    
    binary_mask = np.where(mask_copy >= threshold, 1, 0)

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
    
#------------------------------------------------------------------------------
#                    Dual Formulation of the Total Variation
#------------------------------------------------------------------------------
    
def projB(z) :
    norm = np.linalg.norm(z)
    if norm <= 1 : r=z
    else : r=z/norm
    return r 

def update_z(u, z, sigma, c1, c2) :

    gradxy_u = np.array([gradx(u), grady(u)])
    z_new = np.zeros_like(gradxy_u)  # taille (2, 242, 308)
    for j in range(gradxy_u.shape[1]) :
        for k in range(gradxy_u.shape[2]) : 
            z_new[:,j,k] = projB(z[:,j,k] + sigma*gradxy_u[:,j,k])
    # la projection terme à terme : (z dans R^2)
    # pour chaque pixel on applique projB au couple de valeurs des deux images (gradx, grady)
    return z_new
    
     
def test_dual(mask,z0, image, tau, sigma, lamb, c1, c2, nb_iter, eps, threshold) : 
    u0 = np.copy(mask)
    u = u0
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
        u = projection(u - tau*grad_u)
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
