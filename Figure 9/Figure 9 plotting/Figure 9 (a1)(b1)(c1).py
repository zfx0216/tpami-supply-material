import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

config = {
    "font.family": 'serif',
    "font.size": 15,
    'font.style': 'normal',
    'font.weight': 'normal',
    "mathtext.fontset": 'cm',
    "font.serif": ['cmb10'],
    "axes.unicode_minus": False,
}
plt.rcParams.update(config)

x_1_ALL = [r'$\mathit{f}^{0}(X)$', r'$\mathit{f}^{2500}(X)$', r'$\mathit{f}^{5000}(X)$', r'$\mathit{f}^{7500}(X)$', r'$\mathit{f}^{10000}(X)$']

data_1_ALL = [42.8, 13.2, 13.4, 11.6, 11.6]
data_2_ALL = [42.8, 13.8, 15.4, 20.6, 18.4]
data_3_ALL = [54.8, 19.6, 19.8, 21.4, 19.2]
data_4_ALL = [54.8, 21.2, 23.8, 25, 25.4]

plt.title('the attack success rate on the AlexNet model')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel(' ')
plt.ylabel(' ')

line1, = plt.plot(x_1_ALL, data_1_ALL, markersize=4, linewidth=2.0, color='red', marker='o', linestyle='-')
line2, = plt.plot(x_1_ALL, data_2_ALL, markersize=4, linewidth=2.0, color='blue', marker='o', linestyle='-')
line3, = plt.plot(x_1_ALL, data_3_ALL, markersize=4, linewidth=2.0, color='red', marker='o', linestyle='--')
line4, = plt.plot(x_1_ALL, data_4_ALL, markersize=4, linewidth=2.0, color='blue', marker='o', linestyle='--')

plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.ylim(0, 60)

plt.text(0.23, 44, '88.50%', ha='right', va='center', color='black', fontsize=10)
plt.text(0.23, 56.1, '91.40%', ha='right', va='center', color='black', fontsize=10)

legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', linewidth=2.0, label='PGD on the model trained by adversarial samples with\nreduced background classification contribution value'),
    Line2D([0], [0], color='blue', linestyle='-', linewidth=2.0, label='PGD on the model trained by MI-FGSM adversarial\nexamples'),
    Line2D([0], [0], color='red', linestyle='--', linewidth=2.0, label='C&W on the model trained by adversarial samples with\nreduced background classification contribution value'),
    Line2D([0], [0], color='blue', linestyle='--', linewidth=2.0, label='C&W on the model trained by MI-FGSM adversarial\nexamples')
]

plt.legend(handles=legend_elements, loc='upper right', prop={'size': 10})

plt.savefig('(a1)the attack success rate on the AlexNet model.png', dpi=300)
plt.show()




import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
from matplotlib.lines import Line2D

config = {
    "font.family": 'serif',
    "font.size": 15,
    'font.style': 'normal',
    'font.weight': 'normal',
    "mathtext.fontset": 'cm',
    "font.serif": ['cmb10'],
    "axes.unicode_minus": False,
}
plt.rcParams.update(config)

x_1_ALL = [r'$\mathit{f}^{0}(X)$', r'$\mathit{f}^{2500}(X)$', r'$\mathit{f}^{5000}(X)$', r'$\mathit{f}^{7500}(X)$', r'$\mathit{f}^{10000}(X)$']

data_1_ALL = [20, 11.4, 12.4, 6.8, 7.2]
data_2_ALL = [20, 12, 12.4, 11.2, 13.2]
data_3_ALL = [24.8, 10, 11, 8.2, 10]
data_4_ALL = [24.8, 11.8, 13.4, 9.8, 15.2]

# Plot configuration
plt.title('the attack success rate on the ResNet model')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel(' ')
plt.ylabel(' ')

# Plot lines
line1, = plt.plot(x_1_ALL, data_1_ALL, markersize=4, linewidth=2.0, color='red', linestyle='-', marker='o')
line2, = plt.plot(x_1_ALL, data_2_ALL, markersize=4, linewidth=2.0, color='blue', linestyle='-', marker='o')
line3, = plt.plot(x_1_ALL, data_3_ALL, markersize=4, linewidth=2.0, color='red', linestyle='--', marker='o')
line4, = plt.plot(x_1_ALL, data_4_ALL, markersize=4, linewidth=2.0, color='blue', linestyle='--', marker='o')

plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.ylim(0, 30)
plt.gca().yaxis.set_major_locator(MultipleLocator(5))

plt.text(0.23, 20.7, '85.20%', ha='right', va='center', color='black', fontsize=10)
plt.text(0.23, 25.5, '97.80%', ha='right', va='center', color='black', fontsize=10)

legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', linewidth=2.0,  label='PGD on the model trained by adversarial samples with\nreduced background classification contribution value'),
    Line2D([0], [0], color='blue', linestyle='-', linewidth=2.0, label='PGD on the model trained by MI-FGSM adversarial\nexamples'),
    Line2D([0], [0], color='red', linestyle='--', linewidth=2.0,  label='C&W on the model trained by adversarial samples with\nreduced background classification contribution value'),
    Line2D([0], [0], color='blue', linestyle='--', linewidth=2.0,  label='C&W on the model trained by MI-FGSM adversarial\nexamples')
]

plt.legend(handles=legend_elements, loc='upper right', prop={'size': 10})

plt.savefig('(b1)the attack success rate on the ResNet model.png', dpi=300)
plt.show()



import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import PercentFormatter, MultipleLocator
from matplotlib.lines import Line2D

config = {
    "font.family": 'serif',
    "font.size": 15,
    'font.style': 'normal',
    'font.weight': 'normal',
    "mathtext.fontset": 'cm',
    "font.serif": ['cmb10'],
    "axes.unicode_minus": False,
}
plt.rcParams.update(config)

x_1_ALL = [r'$\mathit{f}^{0}(X)$', r'$\mathit{f}^{2500}(X)$', r'$\mathit{f}^{5000}(X)$', r'$\mathit{f}^{7500}(X)$', r'$\mathit{f}^{10000}(X)$']

data_1_ALL = [22, 9.2, 7.2, 7.6, 6.4]
data_2_ALL = [22, 10, 9.6, 9.6, 13.2]
data_3_ALL = [27, 9.4, 7.8, 8.8, 9.4]
data_4_ALL = [27, 9.6, 12.4, 11.8, 16]

# Plot configuration
plt.title('the attack success rate on the DenseNet model')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel(' ')
plt.ylabel(' ')

# Plot lines
line1, = plt.plot(x_1_ALL, data_1_ALL, markersize=4, linewidth=2.0, color='red', linestyle='-', marker='o')
line2, = plt.plot(x_1_ALL, data_2_ALL, markersize=4, linewidth=2.0, color='blue', linestyle='-', marker='o')
line3, = plt.plot(x_1_ALL, data_3_ALL, markersize=4, linewidth=2.0, color='red', linestyle='--', marker='o')
line4, = plt.plot(x_1_ALL, data_4_ALL, markersize=4, linewidth=2.0, color='blue', linestyle='--', marker='o')

plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.ylim(0, 30)
plt.gca().yaxis.set_major_locator(MultipleLocator(5))

plt.text(0.23, 22.7, '87.25%', ha='right', va='center', color='black', fontsize=10)
plt.text(0.23, 27.7, '95.48%', ha='right', va='center', color='black', fontsize=10)

legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', linewidth=2.0, marker='o', label='PGD on the model trained by adversarial samples with\nreduced background classification contribution value', markersize=0, markeredgecolor='red', markerfacecolor='red'),
    Line2D([0], [0], color='blue', linestyle='-', linewidth=2.0, marker='o', label='PGD on the model trained by MI-FGSM adversarial\nexamples', markersize=0, markeredgecolor='blue', markerfacecolor='blue'),
    Line2D([0], [0], color='red', linestyle='--', linewidth=2.0, marker='o', label='C&W on the model trained by adversarial samples with\nreduced background classification contribution value', markersize=0, markeredgecolor='red', markerfacecolor='red'),
    Line2D([0], [0], color='blue', linestyle='--', linewidth=2.0, marker='o', label='C&W on the model trained by MI-FGSM adversarial\nexamples', markersize=0, markeredgecolor='blue', markerfacecolor='blue')
]

plt.legend(handles=legend_elements, loc='upper right', prop={'size': 10})

plt.savefig('(c1)the attack success rate on the DesneNet model.png', dpi=300)
plt.show()




