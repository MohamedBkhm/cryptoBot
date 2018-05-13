#!/usr/bin/env python3

"""
Ce programme fait des prédictions sur l'évolution des prix du Bitcoin, Dash et Litecoin
basé sur des méthodes d'apprentissage automatique.
"""

__author__     = "Casa de Crypto"
__build__      = "Casa de Crypto"
__copyright__  = "Copyleft 2018 - Casa de Crypto"
__license__    = "GPL"
__title__      = "Machine Learning pour la prediction du cours du Bitcoin"
__version__    = "1.0.0"
__maintainer__ = "Casa de Crypto"
__email__      = "casa-de-crypto@gmail.com"
__status__     = "Production"
__credits__    = "LOUNIS, BOUKHEMKHAM, BIREM, SUKUMAR, GHASAROSSIAN"


import tkinter as tk #Pour la fenetre
import matplotlib.pyplot as plt #Pour tracer le graphique
#Nos differents modeles
from classifier_BTC import ClassifierBTC
from classifier_LTC import ClassifierLTC
from classifier_DASH import ClassifierDASH
from regressor_BTC import RegressorBTC
from regressor_LTC import RegressorLTC
from regressor_DASH import RegressorDASH


#Fonction main
def main():
    app = tk.Tk() #Notre fenetre
    app.title("Prediction du cours des cryptomonnaies par la Casa de Crypto") #Titre de la fenetre
    appicon = tk.PhotoImage(file="appicon.png") #Icon de l'application
    app.iconphoto(app, appicon) #On l'ajoute a notre fenetre
    app.geometry("700x310+300+100") #Taille de notre fenetre
    app.resizable(False, False) #On ne peut pas la redimensionner

    # Frame principal
    frmMain  = tk.Frame(app)
    frmMain.pack(padx=25, pady=25, side="left", anchor="n") #Alignement a gauche

    # Les labels
    tk.Label(frmMain, text="Bitcoin (BTC)  ->", underline=0).grid(row=0, column=0, pady=4, sticky=tk.NW)
    tk.Label(frmMain, text="Litecoin (LTC) ->", underline=0).grid(row=1, column=0, pady=4)
    tk.Label(frmMain, text="Dash (DASH)  ->", underline=0).grid(row=2, column=0, pady=4)

    # Nos Modeles
    # BTC.
    tk.Button(frmMain, text="Classification", underline=0, command = lambda: predict("BTC", "CLA")).grid(row=0, column=1, padx=4, pady=4)
    tk.Button(frmMain, text="Regression", underline=0, command = lambda: predict("BTC", "REG")).grid(row=0, column=3, pady=4, sticky="SE")

    # LTC.
    tk.Button(frmMain, text="Classification", underline=0, command = lambda: predict("LTC", "CLA")).grid(row=1, column=1, padx=4, pady=4)
    tk.Button(frmMain, text="Regression", underline=0, command = lambda: predict("LTC", "REG")).grid(row=1, column=3, pady=4, sticky="SE")

    # DASH.
    tk.Button(frmMain, text="Classification", underline=0, command = lambda: predict("DASH", "CLA")).grid(row=2, column=1, pady=4)
    tk.Button(frmMain, text="Regression", underline=0, command = lambda: predict("DASH", "REG")).grid(row=2, column=3, pady=4, sticky="SE")

    # Boutton pour quitter
    tk.Label(frmMain, text="", underline=0).grid(row=3, column=2)
    tk.Button(frmMain, text="Quitter", width=7, underline=0, command=app.destroy).grid(row=4, column=3, sticky="SE")

    app.mainloop() #Tant que quelque chose ce passe pas on la garde ouverte


#Fonction pour predire le prix d'une cryptomonnaie avec un modele choisis
def predict(coin, model):

    if coin == "BTC" and model == "CLA":
        model = ClassifierBTC()
        model_name = "Classification par Arbre de decision"
    if coin == "BTC" and model == "REG":
        model = RegressorBTC()
        model_name = "Regression par Gradient Boosting"
    if coin == "LTC" and model == "CLA":
        model = ClassifierLTC()
        model_name = "Classification par Gradient Boosting"
    if coin == "LTC" and model == "REG":
        model = RegressorLTC()
        model_name = "Regression par Extra Trees"
    if coin == "DASH" and model == "CLA":
        model = ClassifierDASH()
        model_name = "Classification par les k plus proches voisins"
    if coin == "DASH" and model == "REG":
        model = RegressorDASH()
        model_name = "Regression par Extra Trees"

    plt.subplot(2, 1, 1)
    plt.title("Prediction du prix pour le {} avec un score de {}.\nScore: {:.4f}".format(coin, model_name, model.score))
    plt.plot(model.y_real, label='Valeur réelle')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.plot(model.y_predict, color="green", label='Prédiction')
    plt.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    main()