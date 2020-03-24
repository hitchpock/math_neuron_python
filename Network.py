import scipy.special
import numpy as np


class NeuralNetwork:
    def __init__(self, input_, hidden, output, learning_rate):
        """Инициализация нейрона.

        :param input_: число входных услов
        :type input_: int
        :param hidden: число скрытых узлов
        :type hidden: int
        :param output: число выходных узлов
        :type output: int
        :param learning_rate: скорость обучения
        :type learning_rate: float
        """
        self.input_nodes = input_
        self.hidden_nodes = hidden
        self.output_nodes = output
        self.lr = learning_rate

        # создаем матрицу весов с входного слоя на скрытый слой
        self.wih = np.random.normal(0.0, pow(self.input_nodes, -0.5),
                                    (self.hidden_nodes, self.input_nodes))

        # создаем матрицу весов со скрытого слоя на выходной слой
        self.who = np.random.normal(0.0, pow(self.hidden_nodes, -0.5),
                                    (self.output_nodes, self.hidden_nodes))

        self.activation_func = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        """Метод тренировки нейронной сети.

        :param inputs_list: входной вектор
        :type inputs_list: list
        :param targets_list: целевой вектор
        :type targets_list: list
        """
        # превращаем входной/целевой вектор в матрицу размерности 2
        # и транспонируем
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # создаем скрытый слой путем матричного перемножения весов
        # матрицы входного-скрытого слоя на входной вектор
        # и прогоняем через функцию активации
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        # создаем выходной слой путем матричного перемножения весов
        # матрицы скрытого-выходного слоя на скрытый слой
        # и прогоняем через функцию активации
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        # вычисляем ошибку путем
        # поэлементного вычитания из целевого вектора выходной слой
        output_errors = targets - final_outputs

        # ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = np.dot(self.who.T, output_errors)

        # обновление весов между скрытым и выходным слоем
        self.who += self.lr * np.dot((output_errors * final_outputs *
                                      (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))

        # обновление весов между входным и скртым слоем
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs *
                                      (1.0 - hidden_outputs)),
                                     np.transpose(inputs))

    def query(self, inputs_list):
        """Метод получения предсказания по входному вектору

        :param inputs_list: входной вектор
        :type inputs_list: list
        :return: матрица выходного слоя,
                 показывает на сколько входной вектор похож на каждую цифру
        :rtype: list(list)
        """
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        return final_outputs
