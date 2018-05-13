#!/usr/bin/env python3
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator, ZeroCount


class RegressorDASH:

    def __init__(self):
		#On lit le fichier CSV grace a pd qui contient les memes données que dans la base de donnée
        tpot_data = pd.read_csv('cryptodata.csv', sep=',')
		#On s'interesse que au prix moyen du DASH, de son volume google trend et son sentiment twitter pour notre Variable X
        X = tpot_data[tpot_data["symbol"] == "DASH"][["price_ave", "volume", "google_trend", "twitter_sent"]].values
		#Pour la variable Y on s'interesse seulement a la variable prix
        y = tpot_data[tpot_data["symbol"] == "DASH"][["price"]].values
        training_features, testing_features, training_target, testing_target = \
                                            train_test_split(X, y, random_state=42)

        # Le score sur l'ensemble de formation était: -6.2249531865813035
        exported_pipeline = make_pipeline(
            VarianceThreshold(threshold=0.05),
            ZeroCount(),
            PCA(iterated_power=1, svd_solver="randomized"),
            StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.8500000000000001, tol=0.001)),
            StackingEstimator(estimator=LinearSVR(C=20.0, dual=False, epsilon=0.01, loss="squared_epsilon_insensitive", tol=0.01)),
            ExtraTreesRegressor(bootstrap=False, max_features=0.8, min_samples_leaf=1, min_samples_split=2, n_estimators=100)
        )

        exported_pipeline.fit(training_features, training_target.ravel())
        self.y_predict = exported_pipeline.predict(testing_features)
        self.y_real    = testing_target.ravel()
        self.score     = exported_pipeline.score(testing_features, testing_target)


if __name__ == '__main__':
    model = RegressorDASH()
    print("Score:", model.score)
    print("Prediction:", model.y_predict)
    print("Réalité:", model.y_real)