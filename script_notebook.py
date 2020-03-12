# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt

from Network import NeuralNetwork

from sklearn.model_selection import train_test_split

# %% [markdown]
# ## Установка занчений нейронной сети

# %%
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.2

# %% [markdown]
# ## Создание нейрона

# %%
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# %% [markdown]
# ## Загрузка данных

# %%
seed = 2020
test_size = 0.1

with open("mnist_train.csv", 'r') as f:
    data = f.readlines()

data = data[1:]

train_data, test_data = train_test_split(data, test_size=test_size, random_state=seed)

# %% [markdown]
# ## Предобработка данных
# %% [markdown]
# Отделяем целевую переменную от списка MNIST, оставшуюся строку переводим в массив float и приводим к диапазону (0, 1)

# %%
def preparation(str_matrix):
    f_matrix = []

    targets = []

    for line in str_matrix:
        target = np.zeros(output_nodes) + 0.1
        target[int(line[0])] = 0.99
        targets.append(target)
        line = line.split(',')
        data = (np.asfarray(line[1:]) / 255.0 * 0.99) + 0.1
        f_matrix.append(data)
    
    return f_matrix, targets


# %%
matrix, targets = preparation(train_data)

# %% [markdown]
# ## Тренировка

# %%
ephos = 5

# for e in range(ephos):
#     print(e)
for index, row in enumerate(matrix):
    n.train(row, targets[index])

# %% [markdown]
# # Проверка
# %% [markdown]
# ## Подгтовка тестовых данных

# %%
test_matrix, test_targets = preparation(train_data)


# %%
scorecard = []

for i, row in enumerate(test_matrix):
    output = n.query(row)
    correct_label = np.argmax(test_targets[i])
    predict_label = np.argmax(output)

    if correct_label == predict_label:
        scorecard.append(1)
    else:
        scorecard.append(0)


# %%
scorecard_array = np.array(scorecard)
print("эффективность =", scorecard_array.sum() / scorecard_array.size)

# %% [markdown]
# # Первоначальное задание

# %%
import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt

from Network import NeuralNetwork

from sklearn.model_selection import train_test_split


# %%
input_nodes = 25
hidden_nodes = 15
output_nodes = 10
learning_rate = 0.1


# %%
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


# %%
with open("number.csv", 'r') as f:
    data = f.readlines()

data = data[1:]

train_data = data


# %%
def preparation(str_matrix):
    f_matrix = []

    targets = []

    for line in str_matrix:
        target = np.zeros(output_nodes) + 0.1
        target[int(line[0])] = 0.99
        targets.append(target)
        line = line.split(',')
        data = (np.asfarray(line[1:]) / 1.0 * 0.99) + 0.1
        f_matrix.append(data)
    
    return f_matrix, targets


# %%
matrix, targets = preparation(train_data)


# %%
ephos = 500

for e in range(ephos):

    for index, row in enumerate(matrix):
        n.train(row, targets[index])


# %%
for i in range(10):
    output = n.query(matrix[i])
    correct_label = np.argmax(targets[i])
    predict_label = np.argmax(output)
    print("want - {}, have - {}".format(str(correct_label), str(predict_label)))


# %%


