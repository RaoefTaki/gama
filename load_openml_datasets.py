import numpy as np

from sklearn.datasets._base import load_data, _convert_data_dataframe
from sklearn.utils import Bunch
from os.path import dirname, join


def load_plants_texture(*, return_X_y=False, as_frame=False):
    """Load and return the one-hundred-plants-texture dataset (classification) from OpenML:
    https://www.openml.org/d/1493

    This dataset has 100 classes and 64 features. It has 1599 samples in total.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target: {ndarray, Series} of shape
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        frame: DataFrame of shape
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.


    (data, target) : tuple if ``return_X_y`` is True
    """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'plants_texture.csv')
    csv_filename = join(module_path, 'data', 'plants_texture.csv')

    feature_names = np.array(
        ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17",
         "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "V29", "V30", "V31", "V32", "V33",
         "V34", "V35", "V36", "V37", "V38", "V39", "V40", "V41", "V42", "V43", "V44", "V45", "V46", "V47", "V48", "V49",
         "V50", "V51", "V52", "V53", "V54", "V55", "V56", "V57", "V58", "V59", "V60", "V61", "V62", "V63", "V64"])

    frame = None
    target_columns = ['target', ]
    if as_frame:
        frame, data, target = _convert_data_dataframe("load_plants_texture",
                                                      data,
                                                      target,
                                                      feature_names,
                                                      target_columns)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 frame=frame,
                 target_names=target_names,
                 DESCR=None,
                 feature_names=feature_names,
                 filename=csv_filename)


def load_tuning_svm(*, return_X_y=False, as_frame=False):
    """Load and return the tuning SVMs dataset (classification) from OpenML:
    https://www.openml.org/d/41976

    This dataset has 2 classes and 81 features. It has 156 samples in total.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target: {ndarray, Series} of shape
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        frame: DataFrame of shape
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.


    (data, target) : tuple if ``return_X_y`` is True
    """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'tuning_svm.csv')
    csv_filename = join(module_path, 'data', 'tuning_svm.csv')

    feature_names = np.array(
        ["datasets", "simple.classes", "simple.attributes", "simple.numeric", "simple.nominal", "simple.samples",
         "simple.dimensionality", "simple.numeric_rate", "simple.nominal_rate", "simple.symbols_min",
         "simple.symbols_max", "simple.symbols_mean", "simple.symbols_sd", "simple.symbols_sum",
         "simple.class_prob_min", "simple.class_prob_max", "simple.class_prob_mean", "simple.class_prob_sd",
         "statistical.skewness", "statistical.skewness_prep", "statistical.kurtosis", "statistical.kurtosis_prep",
         "statistical.abs_cor", "statistical.cancor_1", "statistical.fract_1", "inftheo.class_entropy",
         "inftheo.normalized_class_entropy", "inftheo.attribute_entropy", "inftheo.normalized_attribute_entropy",
         "inftheo.joint_entropy", "inftheo.mutual_information", "inftheo.equivalent_attributes",
         "inftheo.noise_signal_ratio", "modelbased.nodes", "modelbased.leaves.nodes_per_attribute",
         "modelbased.nodes_per_instance", "modelbased.leaf_corrobation", "modelbased.level_min", "modelbased.level_max",
         "modelbased.level_mean", "modelbased.level_sd", "modelbased.branch_min", "modelbased.branch_max",
         "modelbased.branch_mean", "modelbased.branch_sd", "modelbased.attribute_min", "modelbased.attribute_max",
         "modelbased.attribute_mean", "modelbased.attribute_sd", "modelbased.NA", "landmarking.naive_bayes",
         "landmarking.stump_min", "landmarking.stump_max", "landmarking.stump_mean", "landmarking.stump_sd",
         "landmarking.stump_min_gain", "landmarking.stump_random", "landmarking.nn_1", "dcomp.f1", "dcomp.f1v",
         "dcomp.f2", "dcomp.f3", "dcomp.f4", "dcomp.l1", "dcomp.l2", "dcomp.l3", "dcomp.n1", "dcomp.n2", "dcomp.n3",
         "dcomp.n4", "dcomp.t1", "dcomp.t2", "cnet.edges", "cnet.degree", "cnet.density", "cnet.maxComp",
         "cnet.closeness", "cnet.betweenness", "cnet.clsCoef", "cnet.hubs", "cnet.avgPath"])

    frame = None
    target_columns = ['target', ]
    if as_frame:
        frame, data, target = _convert_data_dataframe("load_tuning_svm",
                                                      data,
                                                      target,
                                                      feature_names,
                                                      target_columns)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 frame=frame,
                 target_names=target_names,
                 DESCR=None,
                 feature_names=feature_names,
                 filename=csv_filename)


def load_vehicle(*, return_X_y=False, as_frame=False):
    """Load and return the abalone dataset (classification) from OpenML:
    https://www.openml.org/d/183

    This dataset has 4 classes and 18 features. It has 846 samples in total.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target: {ndarray, Series} of shape
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        frame: DataFrame of shape
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.


    (data, target) : tuple if ``return_X_y`` is True
    """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'vehicle.csv')
    csv_filename = join(module_path, 'data', 'vehicle.csv')

    feature_names = np.array(
        ["COMPACTNESS", "CIRCULARITY", "DISTANCE_CIRCULARITY", "RADIUS_RATIO", "PR.AXIS_ASPECT_RATIO",
         "MAX.LENGTH_ASPECT_RATIO", "SCATTER_RATIO", "ELONGATEDNESS", "PR.AXIS_RECTANGULARITY",
         "MAX.LENGTH_RECTANGULARITY", "SCALED_VARIANCE_MAJOR", "SCALED_VARIANCE_MINOR", "SCALED_RADIUS_OF_GYRATION",
         "SKEWNESS_ABOUT_MAJOR", "SKEWNESS_ABOUT_MINOR", "KURTOSIS_ABOUT_MAJOR", "KURTOSIS_ABOUT_MINOR",
         "HOLLOWS_RATIO"])

    frame = None
    target_columns = ['target', ]
    if as_frame:
        frame, data, target = _convert_data_dataframe("load_vehicle",
                                                      data,
                                                      target,
                                                      feature_names,
                                                      target_columns)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 frame=frame,
                 target_names=target_names,
                 DESCR=None,
                 feature_names=feature_names,
                 filename=csv_filename)
