import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def compute_structure_tensor_rgb(image, sigma):
    # initialiser les composants du tenseur de structure
    j11 = np.zeros_like(image, dtype=np.float64)
    j12 = np.zeros_like(image, dtype=np.float64)
    j22 = np.zeros_like(image, dtype=np.float64)

    # Fait chaque canal de l'image (rouge, vert, bleu)
    for c in range(image.shape[2]):
        u_sigma = gaussian_filter(image[:, :, c], sigma=sigma)
        u_sigma_x = np.gradient(u_sigma, axis=0)
        u_sigma_y = np.gradient(u_sigma, axis=1)

        j11[:, :, c] = u_sigma_x ** 2
        j22[:, :, c] = u_sigma_y ** 2
        j12[:, :, c] = u_sigma_x * u_sigma_y

    return j11, j12, j22


def smooth_structure_tensor_rgb(j11, j12, j22, rho):
    # initialiser les composants du tenseur de structure
    j11_smooth = np.zeros_like(j11)
    j12_smooth = np.zeros_like(j12)
    j22_smooth = np.zeros_like(j22)

    # Fait un flou gaussian sur les composants du tenseur de structure sur chaque canal de l'image (rouge, vert, bleu)
    for c in range(j11.shape[2]):
        j11_smooth[:, :, c] = gaussian_filter(j11[:, :, c], sigma=rho)
        j12_smooth[:, :, c] = gaussian_filter(j12[:, :, c], sigma=rho)
        j22_smooth[:, :, c] = gaussian_filter(j22[:, :, c], sigma=rho)

    return j11_smooth, j12_smooth, j22_smooth


def compute_eigenvalues_rgb(j11, j12, j22):
    # initialiser les variables
    mu1 = np.zeros_like(j11)
    mu2 = np.zeros_like(j11)

    # Fait chaque canal de l'image (rouge, vert, bleu) et calcul mu1 et mu2
    for c in range(j11.shape[2]):
        trace = j11[:, :, c] + j22[:, :, c]
        det = j11[:, :, c] * j22[:, :, c] - j12[:, :, c] ** 2
        discriminant = np.sqrt((j11[:, :, c] - j22[:, :, c]) ** 2 + 4 * j12[:, :, c] ** 2)

        mu1[:, :, c] = 0.5 * (trace + discriminant)
        mu2[:, :, c] = 0.5 * (trace - discriminant)

    return mu1, mu2


def coherence_enhancing_diffusivity(mu1, mu2, alpha):
    # initialiser les variables
    lambda1 = np.zeros_like(mu1)
    lambda2 = np.zeros_like(mu2)

    # Fait chaque canal de l'image (rouge, vert, bleu) et calcul lambda1 et lambda2
    for c in range(mu1.shape[2]):
        diff = mu1[:, :, c] - mu2[:, :, c]
        exp_term = np.exp(-1.0 / (diff ** 2 + np.finfo(float).eps))  # Avoid division by zero

        lambda1[:, :, c] = alpha
        lambda2[:, :, c] = alpha + (1 - alpha) * exp_term

        if np.array_equal(mu1[:, :, c], mu2[:, :, c]):
            lambda2[:, :, c] = alpha
        else:
            lambda2[:, :, c] = alpha + (1 - alpha) * exp_term

    return lambda1, lambda2


def weickert_diffusion(image, num_iterations, alpha):
    # faire un flou par composante
    red = cv2.GaussianBlur(image[:, :, 0], (5, 5), 3)
    green = cv2.GaussianBlur(image[:, :, 1], (5, 5), 3)
    blue = cv2.GaussianBlur(image[:, :, 2], (5, 5), 3)

    redModif = red.copy().astype('float64')
    greenModif = green.copy().astype('float64')
    blueModif = blue.copy().astype('float64')

    for k in range(num_iterations):

        j11, j12, j22 = compute_structure_tensor_rgb(image, 1.0)
        j11_smooth, j12_smooth, j22_smooth = smooth_structure_tensor_rgb(j11, j12, j22, 2.0)

        # mu1,2 = 1/2 (trace(J_rho) +- sqrt( trace²(J_rho) - 4det(J_rho) ))
        mu1, mu2 = compute_eigenvalues_rgb(j11_smooth, j12_smooth, j22_smooth)

        # l1 := alpha
        # l2 := alpha                                           if m1 = m2,
        #       alpha + (1 - alpha) exp ( (-C) / (mu1  mu2)²    else
        lambda1, lambda2 = coherence_enhancing_diffusivity(mu1, mu2, alpha)

        # calculer les vecteurs propres (tangente de 2 phi)
        phi = 0.5 * np.arctan2(2 * j12, j11 - j22)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Matrices de vecteurs propres
        w1 = np.stack((cos_phi, sin_phi), axis=-1)
        w2 = np.stack((-sin_phi, cos_phi), axis=-1)

        D11 = lambda1 * cos_phi ** 2 + lambda2 * sin_phi ** 2
        D22 = lambda1 * sin_phi ** 2 + lambda2 * cos_phi ** 2
        D12 = (lambda1 - lambda2) * cos_phi * sin_phi

        # pour chaque pixel hors bordure
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                aR = (red[i + 1, j] - red[i - 1, j]) / 2
                bR = (red[i, j + 1] - red[i, j - 1]) / 2
                aG = (green[i + 1, j] - green[i - 1, j]) / 2
                bG = (green[i, j + 1] - green[i, j - 1]) / 2
                aB = (blue[i + 1, j] - blue[i - 1, j]) / 2
                bB = (blue[i, j + 1] - blue[i, j - 1]) / 2

                # mise a jour des pixels
                redModif[i, j] += D11[i, j, 0] * aR + D12[i, j, 0] * bR
                greenModif[i, j] += D11[i, j, 1] * aG + D12[i, j, 1] * bG
                blueModif[i, j] += D11[i, j, 2] * aB + D12[i, j, 2] * bB

                # mise a jour des canaux
            red = redModif.copy()
            green = greenModif.copy()
            blue = blueModif.copy()
        print(k)

    # résultat final
    filtered = np.stack((redModif, greenModif, blueModif), axis=-1).astype(np.uint8)

    return cv2.addWeighted(image, 0.5, filtered, 0.5, 0)


# Charger l'image
filename_originale = "img/fingerprint.jpg"
image = cv2.imread(filename_originale)

# Paramètres de la diffusion
num_iterations = 5  # Nombre d'itérations de diffusion

# Appliquer la diffusion
result = weickert_diffusion(image, num_iterations, 1)

# Afficher l'image originale et le résultat
cv2.imshow("Image originale", image)
cv2.imshow("Image diffusee", result.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()
