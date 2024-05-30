import math
import sys
from typing import Callable
import cv2
import imageio
import numpy as np
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


def conductance(gradient: np.ndarray, kappa: float, option: float) -> float:
    """
    Calcule la conductance en fonction du gradient et des paramètres donnés.

    Args :
        gradient (float) : Valeur du gradient.
        kappa (float) : Paramètre de conductance.
        option (float) : Option de calcul de conductance. Il y a deux façons de calculer la conductance.

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


def diffusion_anisotrope(image: np.ndarray, dt: float, conductance: Callable[[np.ndarray, float, float], float],
                         kappa: float, option: float) -> np.ndarray:
    """
    Effectue la diffusion anisotrope sur une image en niveaux de gris.

    Args :
        image (numpy.ndarray) : L'image en niveaux de gris.
        dt (float) : Pas de temps pour la diffusion.
        conductance (function) : Fonction de conductance à utiliser.
        kappa (float) : Paramètre de conductance.
        option (float) : Option de calcul de conductance.

    Returns :
        numpy.ndarray : L'image diffusée.
    """
    redOriginal = image[:, :, 0]
    greenOriginal = image[:, :, 1]
    blueOriginal = image[:, :, 2]

    # Initialiser l'image diffusée
    red = image.copy()[:, :, 0]
    green = image.copy()[:, :, 1]
    blue = image.copy()[:, :, 2]

    # Parcourir chaque pixel de l'image sauf les bordures
    # On ne traite pas les bordures, car on impose leur conduction à 0. Ce qui veut dire que le pixel ne change pas
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            # Obtenir les valeurs des pixels voisins
            voisinsR = {
                "N": redOriginal[i, j + 1],  # nord
                "E": redOriginal[i + 1, j],  # est
                "S": redOriginal[i, j - 1],  # sud
                "W": redOriginal[i - 1, j]  # ouest
            }

            voisinsG = {
                "N": greenOriginal[i, j + 1],  # nord
                "E": greenOriginal[i + 1, j],  # est
                "S": greenOriginal[i, j - 1],  # sud
                "W": greenOriginal[i - 1, j]  # ouest
            }

            voisinsB = {
                "N": blueOriginal[i, j + 1],  # nord
                "E": blueOriginal[i + 1, j],  # est
                "S": blueOriginal[i, j - 1],  # sud
                "W": blueOriginal[i - 1, j]  # ouest
            }

            # Calculer les gradients vers les voisins
            # Voir les équations (8) de l'article
            # pour les trois composantes
            gradR_n = voisinsR["N"] - redOriginal[i, j]
            gradR_e = voisinsR["E"] - redOriginal[i, j]
            gradR_s = voisinsR["S"] - redOriginal[i, j]
            gradR_w = voisinsR["W"] - redOriginal[i, j]

            gradG_n = voisinsG["N"] - greenOriginal[i, j]
            gradG_e = voisinsG["E"] - greenOriginal[i, j]
            gradG_s = voisinsG["S"] - greenOriginal[i, j]
            gradG_w = voisinsG["W"] - greenOriginal[i, j]

            gradB_n = voisinsB["N"] - blueOriginal[i, j]
            gradB_e = voisinsB["E"] - blueOriginal[i, j]
            gradB_s = voisinsB["S"] - blueOriginal[i, j]
            gradB_w = voisinsB["W"] - blueOriginal[i, j]

            # Calculer les coefficients de diffusion en utilisant la fonction de conductance
            # Voir les équations (10) de l'article
            # racine carree de la somme des carres

            # sumN = np.sqrt((gradR_n ** 2) + (gradG_n ** 2) + (gradB_n ** 2))
            # sumE = np.sqrt((gradR_e ** 2) + (gradG_e ** 2) + (gradB_e ** 2))
            # sumS = np.sqrt((gradR_s ** 2) + (gradG_s ** 2) + (gradB_s ** 2))
            # sumW = np.sqrt((gradR_w ** 2) + (gradG_w ** 2) + (gradB_w ** 2))

            sumN = abs(gradR_n) + abs(gradG_n) + abs(gradB_n)
            sumE = abs(gradR_e) + abs(gradG_e) + abs(gradB_e)
            sumS = abs(gradR_s) + abs(gradG_s) + abs(gradB_s)
            sumW = abs(gradR_w) + abs(gradG_w) + abs(gradB_w)

            conductance_n = conductance(sumN, kappa, option)
            conductance_e = conductance(sumE, kappa, option)
            conductance_s = conductance(sumS, kappa, option)
            conductance_w = conductance(sumW, kappa, option)

            # Calculer les différences finies
            # Voir l'équation (7) de l'article
            deltaR_n = conductance_n * gradR_n
            deltaR_e = conductance_e * gradR_e
            deltaR_s = conductance_s * gradR_s
            deltaR_w = conductance_w * gradR_w

            deltaG_n = conductance_n * gradG_n
            deltaG_e = conductance_e * gradG_e
            deltaG_s = conductance_s * gradG_s
            deltaG_w = conductance_w * gradG_w

            deltaB_n = conductance_n * gradB_n
            deltaB_e = conductance_e * gradB_e
            deltaB_s = conductance_s * gradB_s
            deltaB_w = conductance_w * gradB_w

            # Mettre à jour la valeur du pixel en fonction des gradients et des coefficients de diffusion
            # Voir l'équation (7) de l'article

            # delta_sum = dt * (delta_n + delta_e + delta_s + delta_w)
            # red[i, j] += delta_sum
            # green[i, j] += delta_sum
            # blue[i, j] += delta_sum

            red[i, j] += dt * (deltaR_n + deltaR_e + deltaR_s + deltaR_w)
            green[i, j] += dt * (deltaG_n + deltaG_e + deltaG_s + deltaG_w)
            blue[i, j] += dt * (deltaB_n + deltaB_e + deltaB_s + deltaB_w)

    return np.stack((red, green, blue), axis=-1)


def main(filename_originale: str, filenames_bruite: list[str], dt: float, iterations: int, kappa: float, option: float):
    """
        Fonction principale pour appliquer la diffusion anisotrope sur des images.

        Args :
            filename_originale (str) : Chemin de l'image originale.
            filenames_bruite (list) : Chemins des images bruitées.
            dt (float) : Pas de temps pour la diffusion.
            iterations (float) : Nombre d'itérations de diffusion.
            kappa (float) : Paramètre de conductance.
            option (float) : Option de calcul de conductance.
    """

    for filename_bruite in filenames_bruite:
        images = []
        # Charger l'image bruitée et la convertir en niveaux de gris
        img = cv2.imread(filename_bruite).astype('float64')

        # Appliquer le flou gaussien pour réduire le bruit
        image_finale = cv2.GaussianBlur(img, (3,3), 5)

        # Liste pour stocker les valeurs de PSNR
        psnr_values = []

        # Diffuser l'image de manière itérative
        for i in range(iterations):
            image_finale = diffusion_anisotrope(image_finale, dt, conductance, kappa, option)

            # Calculer et stocker la valeur de PSNR
            psnr_value = psnr(cv2.imread(filename_originale, 1), image_finale.astype('uint8'))
            psnr_values.append(psnr_value)

            # Convertir l'image de BGR à RGB pour le GIF
            image_rgb = cv2.cvtColor(image_finale.astype('uint8'), cv2.COLOR_BGR2RGB)

            image = image_rgb.copy()
            images.append(
                cv2.putText(
                    image,
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
        imageio.mimsave(f'{filename_bruite}-diffusion-color.gif', images, duration=10, loop=10)

        # Afficher les résultats
        cv2.imshow(f"{filename_bruite} couleur", img.astype('uint8'))
        cv2.imshow(f"{filename_bruite} couleur diffusee", image_finale.astype('uint8'))

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
kappa = 15
option = 2
main(filename_originale, filenames_bruite, dt, iterations, kappa, option)

# rendu final = la semaine prochaine ( le 29/05 )
# le read me : les dependances et comment on les isntalle, expliquer les parties difficiles des articles,
# la ligne de commande pour lancer le truc, des exemples (le plus important).
# les difficules principales, si on a plus de temps, ce qu'on aurait fait.
