from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

from gama import GamaClassifier
from gama.pygmo.algorithms.demo import DEMO
from gama.search_methods import AsyncEA
from gama.search_methods.pygmo_search import PyGMOSearch
import pygmo as pg
from enum import Enum
from load_openml_datasets import load_plants_texture, load_tuning_svm, load_vehicle


class Dataset(Enum):
    BREAST_CANCER = "breast_cancer"
    PLANTS_TEXTURE = "plants_texture"
    TUNING_SVM = "tuning_svm"
    VEHICLE = "vehicle"

def load_dataset(dataset: Dataset = Dataset.BREAST_CANCER, return_X_y: bool = True):
    X_values = None
    y_values = None

    # Load dataset according to argument
    if dataset == Dataset.BREAST_CANCER:
        X_values, y_values = load_breast_cancer(return_X_y=return_X_y)
    elif dataset == Dataset.PLANTS_TEXTURE:
        X_values, y_values = load_plants_texture(return_X_y=return_X_y)
    elif dataset == Dataset.TUNING_SVM:
        X_values, y_values = load_tuning_svm(return_X_y=return_X_y)
    elif dataset == Dataset.VEHICLE:
        X_values, y_values = load_vehicle(return_X_y=return_X_y)
        pass

    # Return the dataset values
    return X_values, y_values

def run_pygmo_algorithm(algorithm, pipeline_max_length, population_size, nr_of_objectives, max_total_time,
                         dataset, X_train, X_test, y_train, y_test, is_multi_class):
    automl = GamaClassifier(search=PyGMOSearch(primitive_set=GamaClassifier().get_pipeline_components(),
                                               used_algorithm=algorithm,
                                               dim=pipeline_max_length,
                                               population_size=population_size,
                                               nr_of_objectives=nr_of_objectives),
                            max_total_time=max_total_time, store="nothing", n_jobs=1)
    print("|----------------------------------------------------|")
    print(algorithm.__name__, ". Dataset=", dataset.value, ". Population size:", population_size, ". Runtime:",
          max_total_time)
    automl.fit(X_train, y_train)

    label_predictions = automl.predict(X_test)
    probability_predictions = automl.predict_proba(X_test)

    print("acc:", accuracy_score(y_test, label_predictions))
    if not is_multi_class:
        print("roc_auc:", roc_auc_score(y_test, label_predictions))
    else:
        print("roc_auc (ovr):", roc_auc_score(y_test, probability_predictions, multi_class="ovr"))
    print("logloss:", log_loss(y_test, probability_predictions))
    print("length:", len(automl.model))
    print("|----------------------------------------------------|")

def run_experiments(dataset: Dataset = Dataset.BREAST_CANCER, population_size: int = 40, pipeline_max_length: int = 5,
                    max_total_time: int = 180, repetitions: int = 1, is_multi_class: bool = False):
    # Load the datasets
    X, y = load_dataset(dataset=dataset, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=0
    )

    " Run the AsyncEA method "
    for repetition in range(repetitions):
        async_ea = GamaClassifier(search=AsyncEA(population_size=population_size), max_total_time=max_total_time,
                                  store="nothing", n_jobs=1, max_pipeline_length=pipeline_max_length)
        print("|----------------------------------------------------|")
        print("ASYNC_EA. Dataset=", dataset.value, ". Population size:", population_size, ". Runtime:", max_total_time)
        async_ea.fit(X_train, y_train)

        label_predictions = async_ea.predict(X_test)
        probability_predictions = async_ea.predict_proba(X_test)

        print("acc:", accuracy_score(y_test, label_predictions))
        if not is_multi_class:
            print("roc_auc:", roc_auc_score(y_test, label_predictions))
        else:
            print("roc_auc (ovr):", roc_auc_score(y_test, probability_predictions, multi_class="ovr"))
        print("logloss:", log_loss(y_test, probability_predictions))
        print("length:", len(async_ea.model))
        print("|----------------------------------------------------|")

    " Run the single objective algorithms with the predetermined values "
    algorithms_list_single_objective = [pg.de, pg.gwo, pg.ihs, pg.pso, pg.bee_colony]
    for repetition in range(repetitions):
        for algorithm in algorithms_list_single_objective:
            nr_of_objectives = 1
            run_pygmo_algorithm(algorithm=algorithm, pipeline_max_length=pipeline_max_length,
                                population_size=population_size, nr_of_objectives=nr_of_objectives,
                                max_total_time=max_total_time, dataset=dataset, X_train=X_train, X_test=X_test,
                                y_train=y_train, y_test=y_test, is_multi_class=is_multi_class)

    " Run the multi objective algorithms with the predetermined values "
    algorithms_list_multi_objective = [DEMO, pg.nsga2, pg.nspso, pg.maco]
    for repetition in range(repetitions):
        for algorithm in algorithms_list_multi_objective:
            nr_of_objectives = 2
            run_pygmo_algorithm(algorithm=algorithm, pipeline_max_length=pipeline_max_length,
                                population_size=population_size, nr_of_objectives=nr_of_objectives,
                                max_total_time=max_total_time, dataset=dataset, X_train=X_train,
                                X_test=X_test,
                                y_train=y_train, y_test=y_test, is_multi_class=is_multi_class)


if __name__ == "__main__":
    _dataset = Dataset.TUNING_SVM
    _population_size = 40  # Close to 50, and multiple of 4 for nsga_2
    _pipeline_max_length = 5  # Short max pipeline length, so works well for single objective methods
    _max_total_time = 180  # Test: 3 mins 10 times, and 10 mins 5 times
    _repetitions = 1

    # Set whether it is a multiclass dataset or not
    _is_multi_class = False
    if _dataset == Dataset.PLANTS_TEXTURE or _dataset == Dataset.VEHICLE:
        _is_multi_class = True

    # Run the experiments
    run_experiments(dataset=_dataset, population_size=_population_size, pipeline_max_length=_pipeline_max_length,
                    max_total_time=_max_total_time, repetitions=_repetitions, is_multi_class=_is_multi_class)
