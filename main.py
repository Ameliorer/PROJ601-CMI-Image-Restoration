import cv2


def diffusion_anisotrope(image, dt, conductance):
    """
    Cette fonction applique une diffusion anisotrope sur une image.

    Args:
        image: L'image d'entrée sous forme de tableau NumPy.
        dt: Pas de temps pour le processus de diffusion.
        conductance: Fonction qui calcule le coefficient de diffusion à chaque pixel.

    Returns:
        L'image diffusée sous forme de tableau NumPy.
    """

    # Initialiser l'image diffusée
    image_diffusee = image.copy()

    # Parcourir chaque pixel de l'image sauf les bordures
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            # Obtenir les valeurs des pixels voisins
            voisins = {
                "N": image[i, j + 1],  # nord
                "E": image[i + 1, j],  # est
                "S": image[i, j - 1],  # sud
                "W": image[i - 1, j]  # ouest
            }

            # Calculer les gradients vers les voisins
            grad_n = voisins["W"] - image[i, j]
            grad_e = voisins["N"] - image[i, j]
            grad_s = voisins["E"] - image[i, j]
            grad_w = voisins["S"] - image[i, j]

            # Calculer les coefficients de diffusion en utilisant la fonction de conductance
            conductance_n = conductance(grad_n)
            conductance_e = conductance(grad_e)
            conductance_s = conductance(grad_s)
            conductance_w = conductance(grad_w)

            # Calculer les différences finies
            delta_n = conductance_n * grad_n
            delta_e = conductance_e * grad_e
            delta_s = conductance_s * grad_s
            delta_w = conductance_w * grad_w

            # Mettre à jour la valeur du pixel en fonction des gradients et des coefficients de diffusion
            image_diffusee[i, j] += dt * (delta_n + delta_e + delta_s + delta_w)

    return image_diffusee


def conductance(gradient):
    """
  Cette fonction calcule le coefficient de diffusion en fonction du gradient.

  Args:
      gradient: La grandeur du gradient de l'image sous forme de tableau NumPy.

  Returns:
      Un tableau NumPy contenant le coefficient de diffusion pour chaque pixel.
  """
    # Définir les paramètres de conductance

    # Calculer le coefficient de diffusion
    c = 1.0 / (1.0 + (gradient / 15) ** 2)

    return c


# Charger l'image et la convertir en niveaux de gris
filename = ("mandrill-g2.png")
img = cv2.imread(filename)
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGrey = imgGrey.astype('float64')

# Appliquer le flou gaussien pour réduire le bruit
gaussian = cv2.GaussianBlur(imgGrey, (5, 5), 0)

derivX = cv2.Sobel(gaussian,ddepth=-1,dx=1,dy=0)
derivY = cv2.Sobel(gaussian,ddepth=-1,dx=0,dy=1)
abs_grad_x = cv2.convertScaleAbs(derivX)
abs_grad_y = cv2.convertScaleAbs(derivY)
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# Définir les paramètres de diffusion
dt = 0.14
iterations = 10  # Nombre d'itérations de diffusion

# Diffuser l'image de manière itérative
image_finale = gaussian
for _ in range(iterations):
    image_finale = diffusion_anisotrope(image_finale, dt, conductance)


derivX = cv2.Sobel(image_finale,ddepth=-1,dx=1,dy=0)
derivY = cv2.Sobel(image_finale,ddepth=-1,dx=0,dy=1)
abs_grad_x = cv2.convertScaleAbs(derivX)
abs_grad_y = cv2.convertScaleAbs(derivY)
gradfinal = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# Afficher les résultats
cv2.imshow("Image originale", imgGrey.astype('uint8'))
cv2.imshow("Image diffusée", image_finale.astype('uint8'))
cv2.imshow("petit Flou Gaussien", (cv2.GaussianBlur(imgGrey, (3,3), 0)).astype('uint8'))
cv2.imshow("grand Flou Gaussien", (cv2.GaussianBlur(imgGrey, (9,9), 0)).astype('uint8'))
cv2.imshow("gradient", grad.astype('uint8'))
cv2.imshow("gradient final", gradfinal.astype('uint8'))

cv2.waitKey(0)
cv2.destroyAllWindows()
