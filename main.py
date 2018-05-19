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

import time
import tkinter as tk #Pour la fenetre
import matplotlib.pyplot as plt #Pour tracer le graphique
import statistics as st #Pour la variance, ecart type ...
from textblob import TextBlob #Libraire pour le traitement du langage naturel,en particulier pour la polarité des tweets.
from Grabber_de_données.core.poloniex import Poloniex #Récuperer nos prix
from Grabber_de_données.core.googletrend import GoogleTrend #Google Trends
from Grabber_de_données.core.twitter import Twitter #Twitter
#Nos differents modeles
from modeles.classifier_BTC import ClassifierBTC
from modeles.classifier_LTC import ClassifierLTC
from modeles.classifier_DASH import ClassifierDASH
from modeles.regressor_BTC import RegressorBTC
from modeles.regressor_LTC import RegressorLTC
from modeles.regressor_DASH import RegressorDASH



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

    # MODELES
    # BTC.
    tk.Button(frmMain, text="Classification", underline=0, command = lambda: chart("BTC", "CLA")).grid(row=0, column=1, padx=4, pady=4)
    tk.Button(frmMain, text="Regression", underline=0, command = lambda: chart("BTC", "REG")).grid(row=0, column=3, pady=4, sticky="SE")
    tk.Button(frmMain, text="Prédire", underline=0, command = lambda: predict(app, "BTC")).grid(row=0, column=4, pady=4, sticky="SE")

    # LTC.
    tk.Button(frmMain, text="Classification", underline=0, command = lambda: chart("LTC", "CLA")).grid(row=1, column=1, padx=4, pady=4)
    tk.Button(frmMain, text="Regression", underline=0, command = lambda: chart("LTC", "REG")).grid(row=1, column=3, pady=4, sticky="SE")
    tk.Button(frmMain, text="Prédire", underline=0, command = lambda: predict(app, "LTC")).grid(row=1, column=4, pady=4, sticky="SE")

    # DASH.
    tk.Button(frmMain, text="Classification", underline=0, command = lambda: chart("DASH", "CLA")).grid(row=2, column=1, pady=4)
    tk.Button(frmMain, text="Regression", underline=0, command = lambda: chart("DASH", "REG")).grid(row=2, column=3, pady=4, sticky="SE")
    tk.Button(frmMain, text="Prédire", underline=0, command = lambda: predict(app, "DASH")).grid(row=2, column=4, pady=4, sticky="SE")

    # Boutton pour quitter...
    tk.Label(frmMain, text="", underline=0).grid(row=3, column=3)
    tk.Button(frmMain, text="Quitter", width=7, underline=0, command=app.destroy).grid(row=4, column=4, sticky="SE")

    app.mainloop() #Tant que quelque chose ce passe pas on la garde ouverte

	
	

#Fonction pour afficher un modele crée qui prends en parametre le coin qu'on veut et son modele
def chart(coin, model):

    if coin == "BTC" and model == "CLA":
        model = ClassifierBTC()
        model_name = "Classification avec la méthode des arbres de décision"
    elif coin == "BTC" and model == "REG":
        model = RegressorBTC()
        model_name = "Regression avec la méthode du Gradient Boosting"
    elif coin == "LTC" and model == "CLA":
        model = ClassifierLTC()
        model_name = "Classification avec la méthode du Gradient Boosting"
    elif coin == "LTC" and model == "REG":
        model = RegressorLTC()
        model_name = "Regression avec la méthode des Extra Trees"
    elif coin == "DASH" and model == "CLA":
        model = ClassifierDASH()
        model_name = "Classification avec la Méthode des k plus proches voisins"
    elif coin == "DASH" and model == "REG":
        model = RegressorDASH()
        model_name = "Regression avec la méthode des Extra Trees"

	#On affiche les graphiques
    plt.subplot(2, 1, 1)
    plt.title("Prediction du prix pour le {} en utilisant la {}.\nScore: {:.2f}%".format(coin, model_name, model.score*100))
    plt.plot(model.y_real, label='Prix réel')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(model.y_predict, color="green", label='Prediction')
    plt.legend(loc='best')
    plt.show()

	
	
	
#Fonction qui va predire en temps réel le prix
def predict(app, coin):
	# Limite de tweet 500
    number_tweets = 500

    if coin == "BTC":
        classifier      = ClassifierBTC()
        regressor       = RegressorBTC()
        classifier_name = "Arbre de decision"
        regressor_name  = "Gradient Boosting"
        name_coin       = "Bitcoin"
    elif coin == "LTC":
        classifier      = ClassifierLTC()
        regressor       = RegressorLTC()
        classifier_name = "Gradient Boosting"
        regressor_name  = "Extra Trees"
        name_coin       = "Litecoin"
    elif coin == "DASH":
        classifier      = ClassifierDASH()
        regressor       = RegressorDASH()
        classifier_name = "Méthode des k plus proches voisins"
        regressor_name  = "Extra Trees"
        name_coin       = "Litecoin"

	# On récupere les prix, trend et sentiment twitter grace a nos API
    data         = Poloniex.returnTicker("USDT_" + coin)
    google_trend = GoogleTrend.get_hit([name_coin, coin])
    twitter_sent = get_twitter_sentiment([name_coin, coin], number_tweets)
	# Prix actuel
    price  = float(data["last"])
	# Volumr
    volume = int(float(data["baseVolume"]))
	#Si la classification vaut 1 cela veut dire que c'est une hausse
    if classifier.predict(price, volume, google_trend, twitter_sent) == 1:
        increased = "EN HAUSSE"
    else:
        increased = "EN BAISSE"

	#Prediction grace a la regression
    predicted_price = regressor.predict(price, volume, google_trend, twitter_sent)
	
	#Nouvelle fenetre pour afficher les predictions
    w_predict = tk.Toplevel(app)
    w_predict.geometry("750x310+250+100")
    w_predict.resizable(width = False, height = False)
    w_predict.title("Prediction pour le {} ({})".format(name_coin, coin))
    w_predict.appicon = tk.PhotoImage(file="appicon.png")

    #Fenetre principale...
    frmMain  = tk.Frame(w_predict)
    frmMain.pack(padx=25, pady=25, anchor="nw")

    # DATE
    tk.Label(frmMain, text= "Date Actuelle:", underline=0, justify = tk.RIGHT,font = ("Helvetica", 12)).grid(row=0, column=0, sticky="NE")
    entry_date = tk.StringVar(value="{}".format(time.strftime("%d-%m-%Y")))
    tk.Entry(frmMain, textvariable=entry_date, width=10, readonlybackground="white",state="readonly", justify = tk.RIGHT, font = ("Helvetica", 12)).grid(row=0, column=1, padx=5, sticky="NW")

    # HEURE
    tk.Label(frmMain, text= "Heure Actuelle:", underline=0, justify = tk.RIGHT,font = ("Helvetica", 12)).grid(row=1, column=0, sticky="NE")
    entry_time = tk.StringVar(value="{}".format(time.strftime("%H:%M:%S")))
    tk.Entry(frmMain, textvariable=entry_time, width=10, readonlybackground="white",state="readonly", justify = tk.RIGHT,font = ("Helvetica", 12)).grid(row=1, column=1, padx=5, pady=5, sticky="NW")

    # PRIX ACTUEL
    tk.Label(frmMain, text= "Prix Actuel:", underline=0, justify = tk.RIGHT,font = ("Helvetica", 12)).grid(row=2, column=0, sticky="NE")
    entry_cprice = tk.StringVar(value="{:.4f} USD".format(price))
    tk.Entry(frmMain, textvariable=entry_cprice, width=15, readonlybackground="white",state="readonly", justify = tk.RIGHT,font = ("Helvetica", 12)).grid(row=2, column=1, padx=5, sticky="NW")

    # CLASSIFICATION
    tk.Label(frmMain, text= "Classification:", underline=1, justify = tk.RIGHT,font = ("Helvetica", 12)).grid(row=3, column=0, sticky="NE")
    entry_classifier = tk.StringVar(value=classifier_name)
    tk.Entry(frmMain, textvariable=entry_classifier, width=15, readonlybackground="white",state="readonly", justify = tk.RIGHT,font = ("Helvetica", 12)).grid(row=3, column=1, padx=5, pady=5, sticky="NW")

    # PREDICTION CLASSIFICATION
    tk.Label(frmMain, text= "Prediction", underline=0, justify = tk.RIGHT, font = ("Helvetica", 12)).grid(row=2, column=2, sticky="SE")
    entry_increased = tk.StringVar(value=increased)
    tk.Entry(frmMain, textvariable=entry_increased, width=15, readonlybackground="white",state="readonly", justify = tk.RIGHT,font = ("Helvetica", 12)).grid(row=3, column=2, pady=5, sticky="NE")

    # CLASSIFICATION PROBABILITE
    tk.Label(frmMain, text= "Probabilité", underline=1, justify = tk.RIGHT, font = ("Helvetica", 12)).grid(row=2, column=3, sticky="SE")
    entry_class_prob = tk.StringVar(value="{:.2f}%".format(classifier.score*100))
    tk.Entry(frmMain, textvariable=entry_class_prob, width=8, readonlybackground="white",state="readonly", justify = tk.RIGHT,font = ("Helvetica", 12)).grid(row=3, column=3, padx=5, pady=5, sticky="NE")

    # REGRESSION
    tk.Label(frmMain, text= "Regression", underline=1, justify = tk.RIGHT,font = ("Helvetica", 12)).grid(row=4, column=0, sticky="NE")
    entry_regressor = tk.StringVar(value=regressor_name)
    tk.Entry(frmMain, textvariable=entry_regressor, width=15, readonlybackground="white",state="readonly", justify = tk.RIGHT,font = ("Helvetica", 12)).grid(row=4, column=1, padx=5, sticky="NW")

    # PREDICTION REGRESSION
    entry_reg_pred = tk.StringVar(value="[{:.2f} - {:.2f}]".format(predicted_price - regressor.stdev, predicted_price + regressor.stdev))
    tk.Entry(frmMain, textvariable=entry_reg_pred, width=15, readonlybackground="white",state="readonly", justify = tk.RIGHT,font = ("Helvetica", 12)).grid(row=4, column=2, sticky="NE")

    # REGRESSION PROBABILITE
    entry_reg_prob = tk.StringVar(value="{:.2f}%".format(regressor.score*100))
    tk.Entry(frmMain, textvariable=entry_reg_prob, width=8, readonlybackground="white",state="readonly", justify = tk.RIGHT,font = ("Helvetica", 12)).grid(row=4, column=3, padx=5, sticky="NE")

    # SENTIMENT TWITTER
    tk.Label(frmMain, text= "Sentiment Twitter:", underline=1, justify = tk.RIGHT,font = ("Helvetica", 12)).grid(row=5, column=0, sticky="NE")
	# Si le sentiment est superieur a 0.20 il est positif si il est entre -0.2 et 0.2 neutre, sinon negatif
    if twitter_sent >= 0.20:
        sentiment = "POSITIF"
    elif twitter_sent <= -0.20:
        sentiment = "NEGATIF"
    else:
        sentiment = "NEUTRE"
    entry_sentiment = tk.StringVar(value=sentiment)
    tk.Entry(frmMain, textvariable=entry_sentiment, width=15, readonlybackground="white",state="readonly", justify = tk.RIGHT,font = ("Helvetica", 12)).grid(row=5, column=1, padx=5, pady=5, sticky="NW")

    # GOOGLE TREND...
    tk.Label(frmMain, text= "Google Trend:", underline=0, width=15, justify = tk.RIGHT, font = ("Helvetica", 12)).grid(row=5, column=2, sticky="NE")
    entry_google = tk.StringVar(value="{}%".format(google_trend))
    tk.Entry(frmMain, textvariable=entry_google, width=8, readonlybackground="white",state="readonly", justify = tk.RIGHT,font = ("Helvetica", 12)).grid(row=5, column=3, padx=5, pady=5, sticky="NE")

    tk.Label(frmMain, text= "", width=1).grid(row=6, column=3)
    tk.Button(frmMain, text="Quitter", width=7, underline=0, command=w_predict.destroy).grid(row=7, column=3, sticky="SE")

	
	
#Fonction qui retourne le sentiment twitter actuel
def get_twitter_sentiment(query, number_tweets):
	#Clé pour nous connecter a l'API Twitter
    consumer_key        = "p1k9CwDKN7R1GmCOMtklolyuR"
    consumer_secret     = "AlgVYob3SUHGImXXr6JM6ANSFNNsMalLsHiaGGf8QyejQOkfFt"
    access_token        = "726262002730057729-NMjwwG1F6NlloG5HXqGMXWbh7si8DsE"
    access_token_secret = "QRNhPGLzlTQQLmuSm2EcGQ1acORqnpf0ZUngGwvkaZRrf"
	#Connexion
    twitter = Twitter(consumer_key, consumer_secret, access_token, access_token_secret)
	#On lance la recherche des termes les plus recent en anglais a propos de notre crypto
    tweets = twitter.api.search(q = query,
                                lang = "en", result_type = 'recent',
                                count = number_tweets)
	#On crée une liste qui contiendra tout les sentiments
    sentiments = list()
	#On parcourt nos tweets
    for tweet in tweets:
        obj       = TextBlob(tweet.text)
		#Polarité qui va de -1 a 1 pour savoir si c'est negatif ou positif
        sentiment = obj.sentiment.polarity
		#On l'ajoute a notre liste
        sentiments.append(sentiment)
	#On retourne la moyenne
    return st.mean(sentiments)


if __name__ == '__main__':
    main()