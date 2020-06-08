import numpy
import pandas
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

LABEL_UNIQUES = {}


def add_ax_margins(ax, x=0.05, y=0.05):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmargin = x * (xlim[1] - xlim[0])
    ymargin = y * (ylim[1] - ylim[0])

    ax.set_xlim(xlim[0] - xmargin, xlim[1] + xmargin)
    ax.set_ylim(ylim[0] - ymargin, ylim[1] + ymargin)


def drop_infrequent(df, column=None, min_appearances=10):
    if column is not None:
        return df[df.groupby(column)[column].transform('count').ge(min_appearances)]
    else:
        result = df
        for column in result.columns:
            result = result[result.groupby(column)[column].transform('count').ge(min_appearances)]
        return result


def factorize(column):
    if column.dtype in [numpy.float64, numpy.float32, numpy.int32, numpy.int64]:
        return column
    else:
        codes, uniques = pandas.factorize(column)
        LABEL_UNIQUES[column.name] = uniques
        return codes
    # return pandas.factorize(column)[0].astype('int32')  # less memory, different results


def normalize(df):
    # scaler = StandardScaler()
    # n = df.shape[0]
    # batch_size = 200_000
    # index = 0
    # while index < n:
    #     partial_size = min(batch_size, n - index)
    #     partial_df = df[index:index+partial_size]
    #     scaler.partial_fit(partial_df)
    #     index += partial_size
    # return scaler.transform(df)  # how to batch .transform() ?
    return pandas.DataFrame(
        StandardScaler().fit_transform(df.values),
        columns=df.columns
    )


def pca(df, n_components=2):
    pca_ = PCA(n_components=n_components)
    data = pca_.fit_transform(df)
    return pandas.DataFrame(data)


def correlate_sort(df, limit: float = None):
    df = df.corr()
    df = df.mask(numpy.tril(numpy.ones(df.shape)).astype(numpy.bool))
    df = df.stack().reset_index()
    df = df.rename(columns={0: 'corr'})
    df = df.sort_values('corr', ascending=False)
    if limit is not None:
        df = df[df['corr'] >= limit]
    return df.reset_index(drop=True)


# region impute
def impute_mean(df, inplace=False):
    return df.fillna(df.mean(), inplace=inplace)


def impute_interpolation(df, inplace=False):
    return df.interpolate(inplace=inplace)


def impute_hotdeck(df, inplace=False):
    return df.fillna(method='ffill', inplace=inplace)


def impute_regression(df):
    mean_filled = impute_mean(df)
    for column in df.columns:
        regression_values = get_linear_regression_values(
            mean_filled.index.values.reshape(-1, 1),
            mean_filled.loc[:, column].values.reshape(-1, 1),
        )
        mean_filled.loc[df[column].isnull(), column] = regression_values[df[column].isnull()]
    return mean_filled


pandas.DataFrame.impute_mean = impute_mean
pandas.DataFrame.impute_interpolation = impute_interpolation
pandas.DataFrame.impute_hotdeck = impute_hotdeck
pandas.DataFrame.impute_regression = impute_regression


# endregion impute


def get_linear_regression_values(x, y):
    regression_model = LinearRegression()
    regression_model.fit(x, y)
    return regression_model.predict(x)
