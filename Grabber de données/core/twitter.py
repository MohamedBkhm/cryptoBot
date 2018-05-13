#!/usr/bin/env python3
"""
Implemente la classe twitter
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


import tweepy #Module pour l'utilisation de twitter

# AIDE: https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets

class Twitter:
    #Initialisation de l'instance qui sera crée
    def __init__(self,  consumer_key, consumer_secret,
                        access_token, access_token_secret):
						
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)


if __name__ == '__main__':
	#Juste pour tester pour le fun
    consumer_key        = "p1k9CwDKN7R1GmCOMtklolyuR"
    consumer_secret     = "AlgVYob3SUHGImXXr6JM6ANSFNNsMalLsHiaGGf8QyejQOkfFt"
    access_token        = "726262002730057729-NMjwwG1F6NlloG5HXqGMXWbh7si8DsE"
    access_token_secret = "QRNhPGLzlTQQLmuSm2EcGQ1acORqnpf0ZUngGwvkaZRrf"

    twitter = Twitter(consumer_key, consumer_secret, access_token, access_token_secret)

    tweets = twitter.api.search(q=["bitcoin", "btc"], lang="en", result_type='recent', count=5)
    
    for tweet in tweets:
        print(tweet.text)
        print("-------------------------------------------------------------")