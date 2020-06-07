from pandas import DataFrame
import seaborn


def plot_clustermap(df: DataFrame, metric='euclidean', method='average',
                    standard_scale=None, z_score=None,
                    cbar_pos=(.02, .8, .05, .18)):
    return seaborn.clustermap(
        df,
        metric=metric,  # correlation >= euclidean
        method=method,  # ward > centroid > average
        standard_scale=standard_scale,
        z_score=z_score,
        cbar_pos=cbar_pos,
    )


def plot_regression(df: DataFrame, x: str, y: str,
                    hue: str = None, col: str = None, row: str = None,
                    x_estimator=None, lowess: bool = False, jitter: float = .03,
                    xticklabels=None, yticklabels=None, hticklabels=None,
                    bottom=None, left=None):
    seaborn.set(color_codes=True)
    regression = seaborn.lmplot(
        data=df,
        x=x, y=y,
        hue=hue,
        col=col, row=row,
        x_estimator=x_estimator,
        lowess=lowess,
        x_jitter=jitter, y_jitter=jitter,
        scatter_kws={'alpha': 0.7}
    ).set_xticklabels(rotation=40, ha='right')\
     .set_yticklabels(rotation=40)
    if xticklabels is not None:
        regression.set_xticklabels(xticklabels)
        regression.set(xticks=range(len(xticklabels)))
    if yticklabels is not None:
        regression.set_yticklabels(yticklabels)
        regression.set(yticks=range(len(yticklabels)))
    if hue is not None and hticklabels is not None:
        for legend_entry, label in zip(regression._legend.texts, hticklabels):
            legend_entry.set_text(label)
    regression.fig.subplots_adjust(bottom=bottom, left=left)
    return regression
