#!/usr/bin/env python3
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, RobustScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator

#Notre classe DASH pour le classifier
class ClassifierDASH:

    def __init__(self):
		#On lit le fichier CSV grace a pd qui contient les memes données que dans la base de donnée
        tpot_data = pd.read_csv('cryptodata.csv', sep=',')
		#On s'interesse que au prix moyen du DASH, de son volume google trend et son sentiment twitter pour notre Variable X
        X = tpot_data[tpot_data["symbol"] == "DASH"][["price_ave", "volume", "google_trend", "twitter_sent"]].values
		#Pour la variable Y on s'interesse seulement au fait qu'il ai augmenter ou dimininuer
        y = tpot_data[tpot_data["symbol"] == "DASH"][["increased"]].values
		#On split avec un etat aleatoire de 42
        training_features, testing_features, training_target, testing_target = \
                            train_test_split(X, y, random_state=42)

        # Le score sur l'ensemble de formation était:0.7091014890632925
        exported_pipeline = make_pipeline(
            StackingEstimator(estimator=LinearSVC(C=0.01, dual=False, loss="squared_hinge", penalty="l2", tol=0.01)),
            RobustScaler(),
            RBFSampler(gamma=1.0),
            StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=13, min_samples_split=13)),
            Binarizer(threshold=0.0),
            KNeighborsClassifier(n_neighbors=17, p=2, weights="uniform")
        )

        exported_pipeline.fit(training_features, training_target.ravel())
        self.y_predict = exported_pipeline.predict(testing_features)
        self.y_real    = testing_target.ravel()
        self.score     = exported_pipeline.score(testing_features, testing_target)


#Fonction principal
if __name__ == '__main__':
	#On crée une nouvelle instance de la classe
    model = ClassifierDASH()
    print("Score:", model.score)
    print("Prediction:", model.y_predict)
    print("Réalité:", model.y_real)