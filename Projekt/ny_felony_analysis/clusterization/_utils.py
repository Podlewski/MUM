from pandas import DataFrame
import seaborn

from utils.common import add_ax_margins, LABEL_UNIQUES


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
                    bottom=None, left=None):
    regression = seaborn.lmplot(
        data=df,
        x=x, y=y,
        hue=hue,
        col=col, row=row,
        x_estimator=x_estimator,
        lowess=lowess,
        x_jitter=jitter, y_jitter=jitter,
        scatter_kws={'alpha': 0.6}
    ).set_xticklabels(rotation=40, ha='right')\
     .set_yticklabels(rotation=40)
    if x in LABEL_UNIQUES:
        regression.set_xticklabels(LABEL_UNIQUES[x])
        regression.set(xticks=range(len(LABEL_UNIQUES[x])))
    if y in LABEL_UNIQUES:
        regression.set_yticklabels(LABEL_UNIQUES[y])
        regression.set(yticks=range(len(LABEL_UNIQUES[y])))
    if hue is not None and hue in LABEL_UNIQUES:
        for legend_entry, label in zip(regression._legend.texts, LABEL_UNIQUES[hue]):
            legend_entry.set_text(label)
    regression.fig.subplots_adjust(bottom=bottom, left=left)
    add_ax_margins(regression.ax)
    return regression
