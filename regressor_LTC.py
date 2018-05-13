#!/usr/bin/env python3
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor


class RegressorLTC:

    def __init__(self):
		#On lit le fichier CSV grace a pd qui contient les memes données que dans la base de donnée
        tpot_data = pd.read_csv('cryptodata.csv', sep=',')
		#On s'interesse que au prix moyen du LTC, de son volume google trend et son sentiment twitter pour notre Variable X
        X = tpot_data[tpot_data["symbol"] == "LTC"][["price_ave", "volume", "google_trend", "twitter_sent"]].values
		#Pour la variable Y on s'interesse seulement a la variable prix
        y = tpot_data[tpot_data["symbol"] == "LTC"][["price"]].values
        training_features, testing_features, training_target, testing_target = \
                                            train_test_split(X, y, random_state=42)
											

        # Le score sur l'ensemble de formation était: -0.5963858827457206
        exported_pipeline = make_pipeline(
            StackingEstimator(estimator=LinearSVR(C=1.0, dual=False, epsilon=0.001, loss="squared_epsilon_insensitive", tol=0.1)),
            StackingEstimator(estimator=LinearSVR(C=0.5, dual=True, epsilon=0.001, loss="epsilon_insensitive", tol=0.1)),
            StackingEstimator(estimator=ElasticNetCV(l1_ratio=1.0, tol=1e-05)),
            StackingEstimator(estimator=LinearSVR(C=0.0001, dual=True, epsilon=1.0, loss="epsilon_insensitive", tol=0.01)),
            StackingEstimator(estimator=XGBRegressor(learning_rate=0.001, max_depth=3, min_child_weight=1, n_estimators=100, nthread=1, subsample=0.05)),
            MinMaxScaler(),
            ExtraTreesRegressor(bootstrap=False, max_features=0.9500000000000001, min_samples_leaf=1, min_samples_split=4, n_estimators=100)
        )

        exported_pipeline.fit(training_features, training_target.ravel())
        self.y_predict = exported_pipeline.predict(testing_features)
        self.y_real    = testing_target.ravel()
        self.score     = exported_pipeline.score(testing_features, testing_target)


if __name__ == '__main__':
    model = RegressorLTC()
    print("Score:", model.score)
    print("Prediction:", model.y_predict)
    print("Réalité:", model.y_real)
