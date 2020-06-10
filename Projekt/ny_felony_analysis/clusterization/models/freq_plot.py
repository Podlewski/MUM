import pandas
from matplotlib import pyplot as plt
from textwrap import fill

data = pandas.read_csv('occurance_frequency.csv', header=0)
xlabels = ['Poranek', 'Popołudnie', 'Wieczór', 'Noc']

plt.grid(alpha=0.2)
print(data)
for index, row, in data.iterrows():
    row_results = []
    print(row)
    row_results.append(row['MORNING'])
    row_results.append(row['AFTERNOON'])
    row_results.append(row['EVENING'])
    row_results.append(row['NIGHT'])

    plt.plot(xlabels, row_results)
    plt.scatter(x=xlabels, y=row_results, label=fill(row['BOROUGH'], 15))

plt.xlabel('Pora dnia')
plt.ylabel('Procent wystąpienia')
plt.title("Wystąpienia przestępst w trakcie dnia")
plt.tight_layout()
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
legend_x = 1
legend_y = 0.5
plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y))
plt.savefig('freq.png', dpi=300)
plt.close()