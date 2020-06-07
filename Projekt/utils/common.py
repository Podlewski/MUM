import numpy
import pandas
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def factorize(column):
    if column.dtype in [numpy.float64, numpy.float32, numpy.int32, numpy.int64]:
        return column
    else:
        return pandas.factorize(column)[0]
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


def correlate_sort(df):
    df = df.corr()
    df = df.mask(numpy.tril(numpy.ones(df.shape)).astype(numpy.bool))
    df = df.stack().reset_index()
    df = df.rename(columns={0: 'corr'})
    df = df.sort_values('corr', ascending=False)
    return df.reset_index(drop=True)


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


def get_linear_regression_values(x, y):
    regression_model = LinearRegression()
    regression_model.fit(x, y)
    return regression_model.predict(x)


pandas.DataFrame.impute_mean = impute_mean
pandas.DataFrame.impute_interpolation = impute_interpolation
pandas.DataFrame.impute_hotdeck = impute_hotdeck
pandas.DataFrame.impute_regression = impute_regression
