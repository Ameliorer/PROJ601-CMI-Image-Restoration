import math
import sys
from typing import Callable
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt


def barre_de_chargement(i: int):
    percent = ("{0:.1f}").format(100 * (i / float(iterations)))
    filled_length = int(50 * i // iterations)
    bar = '█' * filled_length + '-' * (50 - filled_length)
    sys.stdout.write(f'\r |{bar}| {percent}% ')
    sys.stdout.flush()  # Force l'écriture de la ligne


def psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calcule le Peak Signal-to-Noise Ratio (PSNR) entre deux images en niveaux de gris.

    Args :
        original (numpy.ndarray) : L'image originale en niveaux de gris.
        compressed (numpy.ndarray) : L'image modifiée en niveaux de gris.

    Returns :
        int : le PSNR.
    """
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # s'il n'y a pas de bruit
        return 100
    max_pixel = 255.0
    res = 20 * math.log10(max_pixel / math.sqrt(mse))
    return res


def diffusion_anisotrope(image: np.ndarray, dt: float, conductance: Callable[[np.ndarray, float, float], float],
                         kappa: float, option: float) -> np.ndarray:
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
            grad_n = voisins["N"] - image[i, j]
            grad_e = voisins["E"] - image[i, j]
            grad_s = voisins["S"] - image[i, j]
            grad_w = voisins["W"] - image[i, j]

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


def conductance(gradient: np.ndarray, kappa: float, option: float) -> float:
    """
    Calcule la conductance en fonction du gradient et des paramètres donnés.

    Args :
        gradient (float) : Valeur du gradient.
        kappa (float) : Paramètre de conductance.
        option (int) : Option de calcul de conductance. Il y a deux façons de calculer la conductance.

    Returns :
        float : Valeur de conductance calculée.
    """

    if option == 1:
        c = 1.0 / (1.0 + (gradient / kappa) ** 2)
    elif option == 2:
        c = math.exp(-(gradient / kappa) ** 2)
    else:
        c = 1

    return c


def main(filename_originale: str, filenames_bruite: list[str], dt: float, iterations: int, kappa: float, option: float):
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

    for filename_bruite in filenames_bruite:
        images = []
        # Charger l'image bruitée et la convertir en niveaux de gris
        img = cv2.imread(filename_bruite)
        imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgGrey = imgGrey.astype('float64')

        # Appliquer le flou gaussien pour réduire le bruit
        image_finale = cv2.GaussianBlur(imgGrey, (3, 3), 5)

        # Liste pour stocker les valeurs de PSNR
        psnr_values = []

        # Diffuser l'image de manière itérative
        for i in range(iterations):
            image_finale = diffusion_anisotrope(image_finale, dt, conductance, kappa, option)

            # Calculer et stocker la valeur de PSNR
            psnr_value = psnr(cv2.imread(filename_originale, 0), image_finale.astype('uint8'))
            psnr_values.append(psnr_value)

            # Mettre l'image dans un tableau avec le numéro de l'itération
            img = image_finale.copy()
            images.append(
                cv2.putText(
                    img,
                    str(i),
                    (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .7,
                    (0, 0, 0)
                ).astype('uint8'))

            # Afficher une barre de chargement
            barre_de_chargement(i)
        barre_de_chargement(iterations)

        # Sauvegarder les images dans un GIF
        imageio.mimsave(f'{filename_bruite}-diffusion.gif', images, duration=10, loop=20)

        # Afficher les résultats
        cv2.imshow(filename_bruite, imgGrey.astype('uint8'))
        cv2.imshow(f"{filename_bruite} diffusee", image_finale.astype('uint8'))

        # Tracer le graphe du PSNR en fonction des itérations
        plt.plot(range(iterations), psnr_values, marker='o')
        plt.title(f"PSNR en fonction des itérations (option = {option}, dt = {dt}, kappa = {kappa})")
        plt.xlabel("Itérations")
        plt.ylabel("PSNR (dB)")
        plt.xticks(range(iterations))
        plt.show()

        cv2.waitKey(0)
    cv2.destroyAllWindows()


# Valeurs pour les tests
dir = "img/"
filename_originale = f"{dir}femme.jpg"
filenames_bruite = [f"{dir}femme-noisy50.png", f"{dir}femme-noisy80.png", f"{dir}femme-noisy100.png"]

dt = 0.14
iterations = 60
kappa = 5
option = 2
main(filename_originale, filenames_bruite, dt, iterations, kappa, option)
