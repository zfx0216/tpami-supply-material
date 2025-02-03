import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter, MultipleLocator
from matplotlib.lines import Line2D

config = {
    "font.family": 'serif',
    "font.size": 15,
    'font.style': 'normal',
    'font.weight': 'normal',
    "mathtext.fontset": 'cm',
    "font.serif": ['cmb10'],  #
    "axes.unicode_minus": False,
}
plt.rcParams.update(config)

data_1_ALL = [88.4, 87.87, 88.60, 88.33, 89.10]
data_2_ALL = [88.4, 87.70, 86.83, 83.97, 85.47]

x_1_ALL = [r'$\mathit{f}^{0}(X)$', r'$\mathit{f}^{2500}(X)$', r'$\mathit{f}^{5000}(X)$', r'$\mathit{f}^{7500}(X)$', r'$\mathit{f}^{10000}(X)$']

plt.title('the accuracy of the AlexNet model\non the CIFAR-10 testing set')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel(' ')
plt.ylabel(' ')

# Plot the first line
plt.plot(x_1_ALL, data_1_ALL, marker='o', markersize=4, linewidth=2.0, color='red', label='Line 1')

# Plot the second line
plt.plot(x_1_ALL, data_2_ALL, marker='o', markersize=4, linewidth=2.0, color='blue', label='Line 2')

plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.ylim(70, 100)
plt.gca().yaxis.set_major_locator(MultipleLocator(5))

legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', linewidth=2.0, label='the AlexNet model trained by adversarial samples with reduced\nbackground classification contribution value'),
    Line2D([0], [0], color='blue', linestyle='-', linewidth=2.0, label='the AlexNet model trained by MI-FGSM adversarial examples')
]
plt.legend(handles=legend_elements, loc='lower right', prop={'size': 10})

plt.xticks(fontweight='black', fontsize=15)

plt.savefig('(a2)the accuracy of the AlexNet model on the CIFAR-10 testing set.png', dpi=300)
plt.show()




import matplotlib.pyplot as plt
import numpy as np
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

data_1_ALL = [94.17, 94.7, 94.4, 94.37, 93.97]
data_2_ALL = [94.17, 94.23, 93.73, 93.63, 93.23]

x_1_ALL = [r'$\mathit{f}^{0}(X)$', r'$\mathit{f}^{2500}(X)$', r'$\mathit{f}^{5000}(X)$', r'$\mathit{f}^{7500}(X)$', r'$\mathit{f}^{10000}(X)$']

plt.title('the accuracy of the ResNet model\non the CIFAR-10 testing set')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel(' ')
plt.ylabel(' ')

# Plot the first line
plt.plot(x_1_ALL, data_1_ALL, marker='o', markersize=4, linewidth=2.0, color='red', label='Line 1')

# Plot the second line
plt.plot(x_1_ALL, data_2_ALL, marker='o', markersize=4, linewidth=2.0, color='blue', label='Line 2')

plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.ylim(70, 100)
plt.gca().yaxis.set_major_locator(MultipleLocator(5))

legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', linewidth=2.0, label='the ResNet model trained by adversarial samples with reduced\nbackground classification contribution value'),
    Line2D([0], [0], color='blue', linestyle='-', linewidth=2.0, label='the ResNet model trained by MI-FGSM adversarial examples')
]
plt.legend(handles=legend_elements, loc='lower right', prop={'size': 10})

plt.xticks(fontweight='black', fontsize=15)

plt.savefig('(b2)the accuracy of the ResNet model on the CIFAR-10 testing set.png', dpi=300)
plt.show()



import matplotlib.pyplot as plt
import numpy as np
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

data_1_ALL = [94.7, 95.03, 95.6, 94.97, 95.2]
data_2_ALL = [94.7, 94.73, 93.37, 95.37, 94.17]

x_1_ALL = [r'$\mathit{f}^{0}(X)$', r'$\mathit{f}^{2500}(X)$', r'$\mathit{f}^{5000}(X)$', r'$\mathit{f}^{7500}(X)$', r'$\mathit{f}^{10000}(X)$']

plt.title('the accuracy of the DenseNet model\non the CIFAR-10 testing set')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel(' ')
plt.ylabel(' ')

# Plot the first line
plt.plot(x_1_ALL, data_1_ALL, marker='o', markersize=4, linewidth=2.0, color='red', label='Line 1')

# Plot the second line
plt.plot(x_1_ALL, data_2_ALL, marker='o', markersize=4, linewidth=2.0, color='blue', label='Line 2')

plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.ylim(70, 100)
plt.gca().yaxis.set_major_locator(MultipleLocator(5))

legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', linewidth=2.0, label='the DenseNet model trained by adversarial samples with reduced\nbackground classification contribution value'),
    Line2D([0], [0], color='blue', linestyle='-', linewidth=2.0, label='the DenseNet model trained by MI-FGSM adversarial examples')
]
plt.legend(handles=legend_elements, loc='lower right', prop={'size': 10})

plt.xticks(fontweight='black', fontsize=15)

plt.savefig('(c2)the accuracy of the DenseNet model on the CIFAR-10 testing set.png', dpi=300)
plt.show()


