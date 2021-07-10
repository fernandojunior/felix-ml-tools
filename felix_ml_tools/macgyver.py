from typing import List, Dict

import pandas as pd
import numpy as np

from scipy.stats import chi2_contingency
from scipy.stats import chi2

from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

import ppscore as pps

from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.pca import PCA

pd.set_option("max_columns", None)
pd.set_option("max_rows", None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def calc_sample_size(population_size, confidence_level=95, confidence_interval=2):
    """
    Calculate sample size from population given `confidence_level` and `confidence_interval`

    Ref:
    - https://veekaybee.github.io/2015/08/04/how-big-of-a-sample-size-do-you-need/
    - https://www.surveysystem.com/sscalc.htm
    - https://github.com/veekaybee/data/blob/master/samplesize.py
    """
    # supported confidence levels: 50%, 68%, 90%, 95%, and 99%
    confidence_level_zcores = {50: 0.67, 68: 0.99, 90: 1.64, 95: 1.96, 99: 2.57}

    Z = 0.0
    p = 0.5
    e = confidence_interval / 100.0
    N = population_size
    n_0 = 0.0
    n = 0.0

    # find the num std deviations for that confidence level
    Z = confidence_level_zcores[confidence_level]

    if Z == 0.0:
        return -1

    # calc sample size
    n_0 = ((Z ** 2) * p * (1 - p)) / (e ** 2)

    # ajust sample size for finite population
    n = n_0 / (1 + ((n_0 - 1) / float(N)))

    return int(math.ceil(n))


def gen_confusion_matrices(
    data: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    labels: List[str] = None,
) -> Dict[str, Any]:
    """
    Generate different confusion matrices based on `y_true_col` (groud truth) vs `y_pred_col`
    (predictions) columns from `data`.

    Also, generate overall metrics from these matrices
    """
    data = data[data[y_true_col].notnull() & data[y_pred_col].notnull()]

    # raw confusion matrix without normalization
    raw_matrix = pd.DataFrame(
        index=labels,
        columns=labels,
        data=confusion_matrix(data[[y_true_col]], data[[y_pred_col]], labels=labels),
    )

    recall_matrix = (
        pd.DataFrame(
            index=labels,
            columns=labels,
            data=confusion_matrix(
                data[[y_true_col]], data[[y_pred_col]], labels=labels, normalize="true"
            ),
        )
        .apply(lambda x: round(x, 2) * 100)
        .astype(int)
    )

    precision_matrix = (
        pd.DataFrame(
            index=labels,
            columns=labels,
            data=confusion_matrix(
                data[[y_true_col]], data[[y_pred_col]], labels=labels, normalize="pred"
            ),
        )
        .apply(lambda x: round(x, 2) * 100)
        .astype(int)
    )

    # overall metrics
    acc = np.sum(list(np.diag(raw_matrix))) / raw_matrix.sum().sum()
    recall_avg = np.diag(recall_matrix).mean()
    precision_avg = np.diag(precision_matrix).mean()
    f1_score = 2 * (recall_avg * precision_avg) / (recall_avg + precision_avg)

    return {
        "metrics": {
            "acc": acc,
            "recall_avg": recall_avg,
            "precision_avg": precision_avg,
            "f1_score": f1_score,
        },
        "confusion_matrix": {
            "raw": raw_matrix,  # non normalized confusion matrix (counts)
            "recall_matrix": recall_matrix,
            "precision_matrix": precision_matrix,
        },
    }


def split_dataset(df, frac=.3):
    # Spliting into train/test datasets
    test = df.sample(frac=frac)
    train = df.drop(test.index)

    return train, test


def fill_rate(df):
     # Preparing dataframe to analyze features fill
    fill = pd.DataFrame(100*df.count().sort_values()/df.shape[0])
    fill.reset_index(level=0, inplace=True)
    fill.columns = ['Variable','Fill (%)']
    fill['Fill (%)'] = fill['Fill (%)'].astype(int)

    return fill


def extract_time_deltas(df, input_col, prefix=None):
    prefix = prefix if prefix is not None else f'{input_col}_'
    df[f'{prefix}yyyymm'] = df[input_col].dt.strftime('%Y%m')
    df[f'{prefix}year'] = df[input_col].dt.year
    df[f'{prefix}quarter'] = df[input_col].dt.quarter
    df[f'{prefix}month'] = df[input_col].dt.month
    df[f'{prefix}day'] = df[input_col].dt.day

    return df


def split_col_types(df):
    '''
    Split columns from a dataframe by categorical and numeric types
    '''
    categorical_types = ['object', 'datetime64', 'bool']
    numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    return {
        'categorical': df.select_dtypes(include=categorical_types).columns.unique().tolist(),
        'numeric': df.select_dtypes(include=numeric_types).columns.unique().tolist()
    }


def plot_scatter(df, cols=None, target_variable=None, corner=True):
    '''
    Plot Histogram for each column and also scatter plot for each feature pair.
 
    Refs:
    - https://seaborn.pydata.org/generated/seaborn.pairplot.html
    - https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
    '''
    cols = cols if cols else df.columns
    
    if target_variable not in cols:
        cols = cols + [target_variable]

    sns.pairplot(
        df[cols],
        hue=target_variable,
        corner=corner,
        diag_kind = 'kde',
        plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'}
    )


def plot_ppscore_heatmap(ppscores, figsize=(15,15)):
    '''
    Plot PPScore matrix headmap

    https://stackoverflow.com/a/38914112
    https://github.com/8080labs/ppscore/blob/master/examples/titanic_dataset.ipynb
    https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    '''
    ppscores = (ppscores[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore') * 100).astype(int)

    fig, ax = plt.subplots(figsize=figsize)

    ax = sns.heatmap(
        ppscores,
        cmap="Blues",
        linewidths=0.5,
        square=True,
        annot=True,
        fmt='g'
    )

    ax.set_title("PPS matrix (%)")
    ax.set_xlabel("feature")
    ax.set_ylabel("target")

    return ax


def get_ppscore_feature_relevante(ppscores, target_variable, cols=None, upper_threshold=None):
    '''
    PPScore is computed by an DecisionTreeClassifier.
    PPScore is a proxy for Information Gain (entropy)
    '''
    if cols:
        ppscores = ppscores[ppscores['x'].isin(cols)]

    ppscores = (
        ppscores[
            (ppscores['x'] != ppscores['y'])
            & (ppscores['y'] == target_variable)
        ]
        .drop(['y', 'case', 'baseline_score', 'model_score', 'model'], axis=1)
        .sort_values(by='ppscore', ascending=False)
    )

    if upper_threshold:
        ppscores[ppscores['ppscore'] > upper_threshold]
    
    return ppscores


def find_redundant_features(ppscores, target_variable, upper_bound_threshold=0.8):
    '''
    1. Compute async ppscore between two features A and B:
    - ab_ppscore: PPScore from A to B
    - ba_ppscore: PPScore from B to A
    2. Compute ppscore between each feature and target variable Y:
    - ay_ppscore: PPScore from A to Y
    - by_ppscore: PPScore from B to Y
    3. Filter features that are async related given a PPScore upper bound treshould
        - AB ppscore > upper_bound_threshold 
        - BA ppscore > upper_bound_threshold
    '''
    ppscores = ppscores[['x', 'y', 'ppscore']]

    # ppscore from X feture to Y target variable
    xy_scores = (
        ppscores.copy()[(ppscores['x'] != ppscores['y']) & (ppscores['y'] == target_variable)]
        .sort_values(by='ppscore', ascending=False)
    )

    # ppscore from A feature to B feature
    ab_ppscores = ppscores.copy().rename(columns={"x": "A", "y": "B", "ppscore": "AB_ppscore"})

    # ppscore from B feature to A feature
    ba_ppscores = ppscores.copy().rename(columns={"x": "B", "y": "A", "ppscore": "BA_ppscore"})

    async_ppscores = (
        ab_ppscores
        .merge(ba_ppscores, how='left', on=['A', 'B'])
        .merge(xy_scores[['x', 'ppscore']].rename(columns={'x': 'A', 'ppscore': 'AY_ppscore'}), how='left', on=['A'])
        .merge(xy_scores[['x', 'ppscore']].rename(columns={'x': 'B', 'ppscore': 'BY_ppscore'}), how='left', on=['B'])
    )

    async_ppscores = async_ppscores[
        (async_ppscores['A'] != async_ppscores['B'])
        & (async_ppscores['A'] != target_variable)
        & (async_ppscores['B'] != target_variable)
        & (async_ppscores['AB_ppscore'] > upper_bound_threshold)
        & (async_ppscores['BA_ppscore'] > upper_bound_threshold)
    ]

    itered_pairs = []

    output_rows = []

    for index, row in async_ppscores.iterrows():
        if (row['A'], row['B']) in itered_pairs or (row['B'], row['A']) in itered_pairs:
            continue

        output_rows.append(row)
        itered_pairs.append((row['A'], row['B']))

    return pd.DataFrame(output_rows)


def get_ppscore_matrix(df):
    '''
    Compute asymmetric relationships with PPScore

    https://towardsdatascience.com/rip-correlation-introducing-the-predictive-power-score-3d90808b9598
    '''
    return pps.matrix(df)


def has_cat_corr(df, col_a, col_b, alpha = 0.05, normalize='columns'):
    '''
    Check if two cat variables are correlated
    
    If Statistic >= Critical Value: significant result, reject null hypothesis (H0), dependent.
    If Statistic < Critical Value: not significant result, fail to reject null hypothesis (H0),

    https://stats.stackexchange.com/questions/22347/is-chi-squared-always-a-one-sided-test
    https://medium.com/swlh/how-to-run-chi-square-test-in-python-4e9f5d10249d
    https://stackoverflow.com/questions/29901436/is-there-a-pythonic-way-to-do-a-contingency-table-in-pandas
    https://machinelearningmastery.com/chi-squared-test-for-machine-learning/
    '''
    table = pd.crosstab(df[col_a], df[col_b], normalize=normalize)

    print(pd.crosstab(df[col_a], df[col_b]))

    plt.figure(figsize=(10, 10)) 
    ax = sns.heatmap(table, cmap="Blues", linewidths=0.5, square=True, annot=True)

    ax.set_title(f"Contingency Table ({col_b}, {col_a})")

    stat, p, dof, expected = chi2_contingency(table)

    alpha = 0.05

    prob = 1 - alpha

    critical = chi2.ppf(prob, dof)
    print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

    if abs(stat) >= critical:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')

    # interpret p-value
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')

    # p-value > 0.05 means that we do not reject the null hypothesis at 95% level of confidence. The null hypothesis was that X and Y are independent. 
    return p <= alpha


def plot_hist(df, feature_col, target_variable=None):
    cols = [feature_col]
    
    if target_variable:
        cols.append(target_variable) 

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.histplot(
        df[cols],
        x=feature_col,
        hue=target_variable,
        #element="poly",
        kde=True,
    )


def conf_interval(data, num_boot_samples=1000, seed=42, estimator=np.mean):
    '''
    estimator: statistic of interest; Ex: mean, median, variance ...
    
    # https://aegis4048.github.io/comprehensive_confidence_intervals_for_python_developers#t_vs_z
    # https://aegis4048.github.io/comprehensive_confidence_intervals_for_python_developers#t_vs_z
    # https://medium.com/@dhruvb30/practical-statistics-with-python-1-distributions-theorem-and-confidence-intervals-e0d75661d279
    '''
    np.random.seed(seed)
    # arr = data.tolist()

    ####################### Bootstrap #######################
    means = [estimator(data.sample(len(data), replace = True) ) for _ in range(num_boot_samples)]
    #########################################################

    # pd.Series(means).hist()

    # two sided
    intervals = (np.percentile(means, 2.5), np.percentile(means, 97.5))

    fig, ax = plt.subplots(figsize=(10, 4))

    ax = sns.boxplot(x=means, whis=[2.5, 97.5])
    
    ax.set_title("Confidence Interval (alpha=5%)")

    return intervals


def identify_outliers(df, cols=None):
    '''
    Combine three classifiers from PyOD to identify outliers in a dataframe
    https://pyod.readthedocs.io/en/latest/example.html#featured-tutorials
    '''
    cols = cols if cols else df.columns

    clf_knn = KNN()
    clf_knn.fit(df[cols])
    df['is_outlier_knn'] = pd.Series(clf_knn.labels_)

    clf_pca = PCA()
    clf_pca.fit(df[cols])
    df['is_outlier_pca'] = pd.Series(clf_pca.labels_)

    contamination_rate = np.mean([
        len(df['is_outlier_knn'][df['is_outlier_knn'] == 1]) / float(len(df)), 
        len(df['is_outlier_pca'][df['is_outlier_pca'] == 1]) / float(len(df))
    ])

    print(contamination_rate)

    clf_iforest = IForest(contamination=contamination_rate)
    clf_iforest.fit(df[cols])
    df['is_outlier_iforest'] = clf_iforest.labels_

    df['is_outlier'] = (df['is_outlier_knn'] + df['is_outlier_pca'] + df['is_outlier_iforest']) > 1
    df['is_outlier_knn'] = df['is_outlier_knn'] > 0
    df['is_outlier_pca'] = df['is_outlier_pca'] > 0
    df['is_outlier_iforest'] = df['is_outlier_iforest'] > 0

    # get the prediction on the test data
    # y_test_pred = clf.predict(test_data[splited_feature_cols['numeric']])  # outlier labels (0 or 1)
    # y_test_scores = clf.decision_function(test_data[splited_feature_cols['numeric']])  # outlier scores

    return df, [clf_knn, clf_pca, clf_iforest], cols


def get_features_cardinality(df, upper_threshold=1):
    '''
    Select features by cardinality
    '''
    out = df.apply(pd.Series.nunique)
    out = out[out > upper_threshold]
    out = out.reset_index()

    return out.rename(columns={'index': 'feature', 0: 'score'})


def get_features_variance(df, upper_threshold=0):
    '''
    Select features by variance
    '''
    out = train_data[feature_cols].var()
    out = out[out > upper_threshold]
    out = out.reset_index()

    return out.rename(columns={'index': 'feature', 0: 'score'})
