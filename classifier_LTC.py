#!/usr/bin/env python3
import pandas as pd
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator


class ClassifierLTC:

    def __init__(self):
		#On lit le fichier CSV grace a pd qui contient les memes données que dans la base de donnée
        tpot_data = pd.read_csv('cryptodata.csv', sep=',')
		#On s'interesse que au prix moyen du LTC, de son volume google trend et son sentiment twitter pour notre Variable X
        X = tpot_data[tpot_data["symbol"] == "LTC"][["price_ave", "volume", "google_trend", "twitter_sent"]].values
		#Pour la variable Y on s'interesse seulement au fait qu'il ai augmenter ou dimininuer
        y = tpot_data[tpot_data["symbol"] == "LTC"][["increased"]].values
		#On split avec un etat aleatoire de 42
        training_features, testing_features, training_target, testing_target = \
                            train_test_split(X, y, random_state=42)

        # Le score sur l'ensemble de formation était: 0.6037435468950494
        exported_pipeline = make_pipeline(
            PCA(iterated_power=3, svd_solver="randomized"),
            FastICA(tol=0.30000000000000004),
            StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=2, min_samples_leaf=8, min_samples_split=13)),
            GradientBoostingClassifier(learning_rate=0.01, max_depth=3, max_features=0.3, min_samples_leaf=5, min_samples_split=14, n_estimators=100, subsample=1.0)
        )

        exported_pipeline.fit(training_features, training_target.ravel())
        self.y_predict = exported_pipeline.predict(testing_features)
        self.y_real    = testing_target.ravel()
        self.score     = exported_pipeline.score(testing_features, testing_target)


if __name__ == '__main__':
    model = ClassifierLTC()
    print("Score:", model.score)
    print("Prediction:", model.y_predict)
    print("Réalité:", model.y_real)