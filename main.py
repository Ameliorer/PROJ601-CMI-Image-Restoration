import math

import cv2
import imageio


def diffusion_anisotrope(image, dt, conductance, kappa, option):
    """
    Effectue la diffusion anisotrope sur une image en niveaux de gris.

    Args :
        image (numpy.ndarray) : L'image en niveaux de gris.
        dt (float) : Pas de temps pour la diffusion.
        conductance (function) : Fonction de conductance à utiliser.
        kappa (float) : Paramètre de conductance.
        option (int) : Option de calcul de conductance.

    Returns :
        numpy.ndarray : L'image diffusée.
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
    Calcule la conductance en fonction du gradient et des paramètres donnés.

    Args :
        gradient (float) : Valeur du gradient.
        kappa (float) : Paramètre de conductance.
        option (int) : Option de calcul de conductance. Il y a deux façons de calculer la conductance.

    Returns :
        float : Valeur de conductance calculée.
    """

    if option == 2:
        c = math.exp(-(gradient / kappa) ** 2)
    else:
        c = 1.0 / (1.0 + (gradient / kappa) ** 2)

    return c


def main(filename_originale, filenames_bruite, dt, iterations, kappa, option):
    """
        Fonction principale pour appliquer la diffusion anisotrope sur des images.

        Args :
            filename_originale (str) : Chemin de l'image originale.
            filenames_bruite (list) : Chemins des images bruitées.
            dt (float) : Pas de temps pour la diffusion.
            iterations (int) : Nombre d'itérations de diffusion.
            kappa (float) : Paramètre de conductance.
            option (int) : Option de calcul de conductance.
    """

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
        #imageio.mimsave('diffusion.gif', images)

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
