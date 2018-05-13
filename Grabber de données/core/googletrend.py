#!/usr/bin/env python3
"""
Implementation de la classe GoogleTrend
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


import pytrends.request

class GoogleTrend:
    
    @staticmethod
    def get_hit(kw_list):
        """
        @kw_list: liste des termes de recherche.
        """
        key = kw_list[0]
        GT  = pytrends.request.TrendReq()
        GT.build_payload(kw_list, timeframe = 'now 1-d')
        return GT.interest_over_time()[key][-1]

#Fonction principale
if __name__ == '__main__':
    print(GoogleTrend.get_hit(["bitcoin"]))