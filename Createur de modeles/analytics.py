#!/usr/bin/env python3
"""
Script permettant de crée nos modeles
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

import sqlite3 #SQLITE pour la base de donnée
import pandas as pd #Lire dans la base  de donnée
import multiprocessing #Pour effectuer la creation de nos modeles en même temps grace au multi tasking
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier, TPOTRegressor



#Fonction qui crée le model pour un type d'apprentissage supervisé donner
def models_building(type_):
    """
    @type_ = Classification ou regression
    """
    conn   = sqlite3.connect("cryptodata.db") #Connexion a la bdd
    cursor = conn.cursor()
    coins  = cursor.execute("SELECT symbol FROM crypto;") #On choisis la table
	#On parcourt cette table
    for symbol, in coins:
		#On lit les données
        data = pd.read_sql("SELECT price,  price_ave, increased, volume, google_trend, twitter_sent FROM logbook WHERE symbol='{}';".format(symbol), conn)
		#Notre variable X
        X = data[["price_ave", "volume", "google_trend", "twitter_sent"]]
		#Si c'est un classifier notre variable Y sera l'augmentation ou la diminution 0 ou 1.
        if type_ == "classifier":
            y = data[["increased"]]
		#Sinon si c'est un regresseur ce sera le prix.
        elif type_ == "regressor":
            y = data[["price"]]

		#4 set de données, les deux premiers sont les features pour les training et testing data et les deux derniers sont les labels.
		#Divisenos données en sous-ensembles de tests et de trains aléatoires
		#le training representera 80% de nos données et le testing 20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20)
        if type_ == "classifier":
            model = TPOTClassifier
        elif type_ == "regressor":
            model = TPOTRegressor
		#Verbosity a 3
        tpot = model(verbosity = 3)
		#On fit le model
        tpot.fit(X_train, y_train.values.ravel())

		#On export notre fichier
        if type_ == "classifier":
            tpot.export('classifier_{}.py'.format(symbol))
        elif type_ == "regressor":
            tpot.export('regressor_{}.py'.format(symbol))


#Fonction main
if __name__ == '__main__':
	#Pour le classifieur on crée notre multi tasking pour pouvoir crée le modele des 3 cryptomonnaies en meme temps pour la classification
    classifier = multiprocessing.Process(target = models_building, args=("classifier",))
	#Pareil pour la regression
    regressor = multiprocessing.Process(target = models_building, args=("regressor",))
	#On lance le tout
    classifier.start()
    regressor.start()