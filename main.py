import math

import cv2
import imageio


def diffusion_anisotrope(image, dt, conductance, kappa, option):
    """
    Cette fonction applique une diffusion anisotrope sur une image.

    Args :
        image : L'image d'entrée sous forme de tableau NumPy.
        dt : Pas de temps pour le processus de diffusion.
        conductance : Fonction qui calcule le coefficient de diffusion à chaque pixel.

    Returns :
        L'image diffusée sous forme de tableau NumPy.
    """

    # Initialiser l'image diffusée
    image_diffusee = image.copy()

    # Parcourir chaque pixel de l'image sauf les bordures
    # On ne traite pas les bordures, car on impose leur conduction à 0. Ce qui veut dire que le pixel ne change pas
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
            # Voir les équations (8) de l'article
            grad_n = voisins["W"] - image[i, j]
            grad_e = voisins["N"] - image[i, j]
            grad_s = voisins["E"] - image[i, j]
            grad_w = voisins["S"] - image[i, j]

            # Calculer les coefficients de diffusion en utilisant la fonction de conductance
            # Voir les équations (10) de l'article
            conductance_n = conductance(abs(grad_n), kappa, option)
            conductance_e = conductance(abs(grad_e), kappa, option)
            conductance_s = conductance(abs(grad_s), kappa, option)
            conductance_w = conductance(abs(grad_w), kappa, option)

            # Calculer les différences finies
            # Voir l'équation (7) de l'article
            delta_n = conductance_n * grad_n
            delta_e = conductance_e * grad_e
            delta_s = conductance_s * grad_s
            delta_w = conductance_w * grad_w

            # Mettre à jour la valeur du pixel en fonction des gradients et des coefficients de diffusion
            # Voir l'équation (7) de l'article
            image_diffusee[i, j] += dt * (delta_n + delta_e + delta_s + delta_w)

    return image_diffusee


def conductance(gradient, kappa, option):
    """
    Cette fonction calcule le coefficient de diffusion en fonction du gradient.

    Args :
        gradient : La grandeur du gradient de l'image sous forme de tableau NumPy.
        kappa :
        option :

        Les espaces d'échelle générés par ces deux fonctions sont différents :
        le premier privilégie les arêtes à fort contraste par rapport à celles à faible contraste,
        tandis que le second privilégie les régions larges par rapport aux plus petites.

    Returns :
        Un tableau NumPy contenant le coefficient de diffusion pour chaque pixel.
    """

    if option == 2:
        c = math.exp(-(gradient / kappa) ** 2)
    else:
        # Calculer le coefficient de diffusion
        c = 1.0 / (1.0 + (gradient / kappa) ** 2)

    return c


def main(filename_originale, filenames_bruite, dt, iterations, kappa, option):
    # Charger l'image originale et la convertir en niveaux de gris
    img_originale = cv2.imread(filename_originale)
    img_originale = cv2.cvtColor(img_originale, cv2.COLOR_BGR2GRAY)
    img_originale = img_originale.astype('float64')
    cv2.imshow("Image originale", img_originale.astype('uint8'))
    cv2.waitKey(0)

    images = []

    for filename_bruite in filenames_bruite:
        # Charger l'image bruitée et la convertir en niveaux de gris
        img = cv2.imread(filename_bruite)
        imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgGrey = imgGrey.astype('float64')

        # Appliquer le flou gaussien pour réduire le bruit
        gaussian = cv2.GaussianBlur(imgGrey, (5, 5), 0)

        # Diffuser l'image de manière itérative
        image_finale = gaussian
        for _ in range(iterations):
            image_finale = diffusion_anisotrope(image_finale, dt, conductance, kappa, option)
            images.append(image_finale.astype('uint8'))

        # Sauvegarder les images dans un GIF
        imageio.mimsave('diffusion.gif', images)

        # Afficher les résultats
        cv2.imshow(filename_bruite, imgGrey.astype('uint8'))
        cv2.imshow(f"{filename_bruite} diffusée", image_finale.astype('uint8'))
        cv2.waitKey(0)
    cv2.destroyAllWindows()


# Valeurs pour les tests
filename_originale = "IRM.jpeg"
filenames_bruite = ["IRM.jpeg"]
dt = 0.14
iterations = 100
kappa = 5
option = 2
main(filename_originale, filenames_bruite, dt, iterations, kappa, option)
