from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

from gama import GamaClassifier
from gama.search_methods import AsynchronousSuccessiveHalving
from gama.search_methods import AsyncEA
from gama.search_methods import RandomSearch
from gama.search_methods import DifferentialEvolution
from gama.utilities.metrics import Metric

if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=0
    )

    pipeline_max_length = 5
    population_size = 40

    # Fit DE classifier
    automl = GamaClassifier(search=DifferentialEvolution(primitive_set=GamaClassifier().get_pipeline_components(),
                                                         dim=pipeline_max_length,
                                                         population_size=population_size),
                            max_total_time=180, store="nothing", n_jobs=1)
    print("Starting `fit` which will take roughly 3 minutes.")
    automl.fit(X_train, y_train)

    label_predictions = automl.predict(X_test)
    probability_predictions = automl.predict_proba(X_test)

    print("acc:", accuracy_score(y_test, label_predictions))
    print("roc_auc:", roc_auc_score(y_test, label_predictions))
    print("logloss:", log_loss(y_test, probability_predictions))
    print("length:", len(automl.model))
