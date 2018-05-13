#!/usr/bin/env python3
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tpot.builtins import StackingEstimator


class RegressorBTC:

    def __init__(self):
		#On lit le fichier CSV grace a pd qui contient les memes données que dans la base de donnée
        tpot_data = pd.read_csv('cryptodata.csv', sep=',')
		#On s'interesse que au prix moyen du BTC, de son volume google trend et son sentiment twitter pour notre Variable X
        X = tpot_data[tpot_data["symbol"] == "BTC"][["price_ave", "volume", "google_trend", "twitter_sent"]].values
		#Pour la variable Y on s'interesse seulement a la variable prix
        y = tpot_data[tpot_data["symbol"] == "BTC"][["price"]].values
        training_features, testing_features, training_target, testing_target = \
                                            train_test_split(X, y, random_state=42)

        # Le score sur l'ensemble de formation était: -1110.2102803031512
        exported_pipeline = make_pipeline(
            make_union(
                FeatureAgglomeration(affinity="l1", linkage="complete"),
                make_pipeline(
                    SelectPercentile(score_func=f_regression, percentile=38),
                    RobustScaler(),
                    MinMaxScaler()
                )
            ),
            StackingEstimator(estimator=ElasticNetCV(l1_ratio=1.0, tol=0.01)),
            StackingEstimator(estimator=RidgeCV()),
            GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="ls", max_depth=9, max_features=0.45, min_samples_leaf=17, min_samples_split=16, n_estimators=100, subsample=0.6500000000000001)
        )

        exported_pipeline.fit(training_features, training_target.ravel())
        self.y_predict = exported_pipeline.predict(testing_features)
        self.y_real    = testing_target.ravel()
        self.score     = exported_pipeline.score(testing_features, testing_target)


if __name__ == '__main__':
    model = RegressorBTC()
    print("Score:", model.score)
    print("Prediction:", model.y_predict)
    print("Réalité:", model.y_real)