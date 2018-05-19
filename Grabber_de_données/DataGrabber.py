#!/usr/bin/env python3
"""
Script permettant de remplir notre base de données
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

import sys
sys.path.append('../')

import multiprocessing #Multi Tache
import statistics as st
import time #Temps
from textblob import TextBlob #traitement de données textuelles
from core.poloniex import Poloniex #API Poloniex
from core.googletrend import GoogleTrend #API Google Trend
from core.twitter import Twitter # API Twitter
from cryptodata import Cryptodata

#Fonction pour crée la base de donnée
def create_database():
    database = Cryptodata()
    database.create()
	#On crée un dictionnaire que l'on remplira avec ce qu'on veut
    crypto = dict()
    crypto["symbol"] = "BTC"
    crypto["name"]   = "Bitcoin"
    database.insert_crypto(crypto)

    crypto["symbol"] = "DASH"
    crypto["name"]   = "Dash"
    database.insert_crypto(crypto)

    crypto["symbol"] = "LTC"
    crypto["name"]   = "Litecoin"
	#On insert le dictionnaire dans la base table
    database.insert_crypto(crypto)


#Fonction qui permet d'obtenir le sentiment twitter avec comme parametre le terme que l'on veut chercher et le nombre de tweets
def get_twitter_sentiment(query, number_tweets):
	#Nos identifiants twitter pour se connecter a l'API
    consumer_key        = "p1k9CwDKN7R1GmCOMtklolyuR"
    consumer_secret     = "AlgVYob3SUHGImXXr6JM6ANSFNNsMalLsHiaGGf8QyejQOkfFt"
    access_token        = "726262002730057729-NMjwwG1F6NlloG5HXqGMXWbh7si8DsE"
    access_token_secret = "QRNhPGLzlTQQLmuSm2EcGQ1acORqnpf0ZUngGwvkaZRrf"
    #Connexion a l'API
    twitter = Twitter(consumer_key, consumer_secret, access_token, access_token_secret)
	#On lance notre recherche, on veut les X nombres de tweet les plus recents pour notre terme
    tweets = twitter.api.search(q = query,
                                lang = "en", result_type = 'recent',
                                count = number_tweets)
	#On enregistre les sentiments dans une liste
    sentiments = list()
	#On parcourt les tweets
    for tweet in tweets:
        obj       = TextBlob(tweet.text)
        sentiment = obj.sentiment.polarity #Polarité qui va de -1 a 1 pour savoir si c'est negatif ou positif
        sentiments.append(sentiment) #On ajoute le sentiment au notre liste

    return st.mean(sentiments) #On retourne la moyenne de tout nos sentiments



#Fonction qui permet d'obtenir la moyenne du prix
def moyenne_prix(list_logs):
    acum = 0
    n    = 0
    for item in list_logs:
        n += 1
        acum += item["price"]
    return acum/n


def remplir_bdd():
	#On se limite a 200 tweets
    number_tweets     = 200
	#7 prix pour la moyenne
    number_prices     = 7
	#Toute les 9 minutes car on a une limitation dans l'utilisation des API twitter et google
    run_every_minutes = 9
	
    database = Cryptodata()
	#List qui va logs tout les coins deja parcouru
    logs     = dict()
    i        = 0
	#Tant que c'est vrai on tourne
    while True:
        i += 1
        print("\n* Lancement de la session {} le {}.".format(i, time.strftime("%H:%M:%S")))
        
        cryptos = database.execute("SELECT symbol, name FROM crypto;")
		#On le fait pour chaque crypto
        for symbol, name in cryptos:
            print("    > Recherche {}...".format(name))
			#Si on ne l'a deja pas parcouru
            if symbol not in logs:
				#On l'ajoute
                logs[symbol] = list()
			#on lance nos recherches
            data         = Poloniex.returnTicker("USDT_" + symbol)
            google_trend = GoogleTrend.get_hit([name, symbol])
            twitter_sent = get_twitter_sentiment([name, symbol], number_tweets)
			
            if len(logs[symbol]) > 0:
				#On calcul la moyenne 
                price_ave = moyenne_prix(logs[symbol])
				#On set a 1 si ca a augmenter et 0 si ca a diminuer
                increased = "1" if float(data["last"]) > logs[symbol][-1]["price"] else "0"
            else:
				#Sinon on set a 0
                price_ave = 0
                increased = "0"
			#Dictionnaire qui va stocker toute nos données
            log = dict()
            log["symbol"]       = symbol
            log["date_log"]     = time.strftime("%d-%m-%Y") #Date actuelle
            log["time_log"]     = time.strftime("%H:%M:%S") #Heure actuelle
            log["price"]        = float(data["last"]) #Notre prix est le dernier de la liste
            log["price_ave"]    = price_ave #moyenne
            log["increased"]    = increased #augmentation
            log["volume"]       = int(float(data["baseVolume"])) #Volume
            log["google_trend"] = int(google_trend) #Score google trend
            log["twitter_sent"] = twitter_sent #Sentiment Twitter
            logs[symbol].append(log)  #On ajoute au dictionnaire
			
            if len(logs[symbol]) == number_prices + 1:
                logs[symbol].pop(0)
                item              = logs[symbol][-2].copy()
                item["price"]     = logs[symbol][-1]["price"]
                item["increased"] = logs[symbol][-1]["increased"]
                database.insert_logbook(item)

        print("Fin de la recherche a {}. Reprise dans {} minutes.".format(time.strftime("%H:%M:%S"), run_every_minutes))
		
		#Pause de 9 minutes
        time.sleep(run_every_minutes*60)


if __name__ == '__main__':
	#Cree la bdd
    create_database()
	#On la remplit
    process = multiprocessing.Process(target = remplir_bdd)
	#Tant que c'est vrai et que le processe n'est pas lancer on lance
    while True:
        if not process.is_alive():
            print("MAIN: remplir_bdd() n'est pas lancer. Relance en cours...")
            process.start()