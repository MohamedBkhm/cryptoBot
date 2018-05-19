#!/usr/bin/env python3
"""
Implemente la classe Poloniex
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

import json #Lire le resultats de la recherche
import urllib.request #Pour lancer nos requetes


class Poloniex:

    URL_BASE = 'https://poloniex.com/public' #ULR de base

	#Fonction qui lance une requete et decode le JSON
    @staticmethod
    def get_json(url):
        try:
            json_data = urllib.request.urlopen(url)
        except:
            raise
        else:
            return json.loads(json_data.read().decode("utf-8"))

    @staticmethod
	#Rtourne le ticker pour une crypto donnée
    def returnTicker(currencyPair = None):
        """"""
        url       = Poloniex.URL_BASE + "?command=returnTicker"
        data      = Poloniex.get_json(url)
        if currencyPair:
            return data[currencyPair]
        else:
            return data


if __name__ == '__main__':
    print(Poloniex.returnTicker("USDT_BTC"))