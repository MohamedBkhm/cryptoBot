#!/usr/bin/env python3
"""
Gère la persistance des données dans SQLite.
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


import os
import sqlitedb #Lib pour sqlite


class Cryptodata(sqlitedb.SQLiteDB):
	#Notre DB
    __database = "cryptodata.db"


    def __init__(self):
        super().__init__(Cryptodata.__database)

	#Fonction de creation
    def create(self):
        os.remove(Cryptodata.__database)
        self.setDatabase(Cryptodata.__database)
        sql = """
        CREATE TABLE crypto (
            symbol VARCHAR(10) NOT NULL,
            name VARCHAR(20) NOT NULL,
            PRIMARY KEY (symbol)
        );

        CREATE TABLE logbook (
            id_log INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol VARCHAR(10) NOT NULL,
            date_log VARCHAR(10) NOT NULL,
            time_log VARCHAR(8) NOT NULL,
            price REAL NOT NULL,
            price_ave REAL NOT NULL,
            increased VARCHAR(1) NOT NULL,
            volume BIGINT UNSIGNED NOT NULL,
            google_trend TINYINT UNSIGNED NOT NULL,
            twitter_sent REAL NOT NULL,
            FOREIGN KEY (symbol) REFERENCES crypto(symbol) ON UPDATE CASCADE ON DELETE RESTRICT
        );

        CREATE INDEX i_logbook_symbol ON logbook(symbol);
        """
		#On parcourt les differentes requetes
        for stm in sql.split(";"):
			#On les executes en supprimant en rajoutant le point virgule que l'on a enlever pour les parcourir
            self.execute(stm + ";")

        return True

	#Fonction qui permet d'inserer une crypto
    def insert_crypto(self, crypto):
        sql = """INSERT INTO crypto (symbol, name) VALUES ("{symbol}", "{name}");""".format(**crypto)
        self.execute(sql)
	
	#Fonction qui permet de rajouter des logs
    def insert_logbook(self, log):
        sql = """INSERT INTO logbook (symbol, date_log, time_log, price, price_ave, increased, volume, google_trend, twitter_sent) VALUES ("{symbol}", "{date_log}", "{time_log}", {price}, {price_ave}, "{increased}", {volume}, {google_trend}, {twitter_sent});""".format(**log)

        self.execute(sql)


if __name__ == '__main__':
	#On test
    database = Cryptodata()
    database.create()

    crypto = dict()
    crypto["symbol"] = "BTC"
    crypto["name"]   = "Bitcoin"
    database.insert_crypto(crypto)

    crypto["symbol"] = "DASH"
    crypto["name"]   = "Dash"
    database.insert_crypto(crypto)

    crypto["symbol"] = "LTC"
    crypto["name"]   = "Litecoin"
    database.insert_crypto(crypto)

    log = dict()
    log["symbol"]       = "BTC"
    log["date_log"]     = "X"
    log["time_log"]     = "X"
    log["price"]        = 10.10
    log["price_ave"]    = 20
    log["increased"]    = "1"
    log["volume"]       = 30
    log["google_trend"] = 40
    log["twitter_sent"] = 50

    database.insert_logbook(log)