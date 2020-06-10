import pandas as pd
from matplotlib import pyplot as plt
from textwrap import fill

files = ['Klasyfikator', 'Cecha', 'UsunieteCechy']
charts = ['Klasyfikatory', 'Cechy', 'Usunięte Cechy']
xlabels = ['Średnia Precyzja', 'Średnia Dokładność', 'Średnia Czułość']

for file_name, charts_names in zip(files, charts):
    classifiers = pd.read_csv(file_name + '.csv', sep=';', header=0)

    for index, row, in classifiers.iterrows():
        row_results = []
        row_results.append(row['Średnia Precyzja'])
        row_results.append(row['Średnia Dokładność'])
        row_results.append(row['Średnia Czułość'])
        
        plt.plot(xlabels, row_results, zorder=3)
        plt.scatter(x=xlabels, y=row_results, label=fill(row['Nazwa'], 15), zorder=6)

    plt.grid(alpha=0.2, zorder=0)
    plt.xlabel('Metryka')
    plt.ylabel('Wartość')
    plt.title(charts_names)
    plt.tight_layout()
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
    legend_x = 1
    legend_y = 0.5
    plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y))
    plt.savefig(file_name + '.png', dpi=300)
    plt.close()


    i = 0
    clas = len(classifiers.index)
    for index, row, in classifiers.iterrows():
        plt.bar(i * clas, row['Średni Czas'], label=fill(row['Nazwa'], 15), zorder=3)
        i = i + 0.15

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.grid(alpha=0.2, zorder=0)
    plt.xlabel('Średni czas')
    plt.ylabel('Czas (s)')
    plt.title(charts_names)
    plt.tight_layout()
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.65, box.height])
    legend_x = 1
    legend_y = 0.5
    plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y))
    plt.savefig(file_name + '_Czas.png', dpi=300)
    plt.close()

    