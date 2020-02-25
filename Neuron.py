import numpy as np


class Neuron:
    """
    Класс нейрона или перцептрон.
    """

    def __init__(self, weight, bias):
        """
        Метод инициализации класса.

        :param weight: веса перцептрона
        :type weight: list(list())
        :param bias: смещение
        :type bias: int, float
        """
        self.weight = np.array(weight)
        self.bias = bias

    def compute_with_bias(self, x):
        """
        Вычисление с помощью смещения.

        :param x: список входных параметров
        :type x: list
        :return: выход перцептрона
        :rtype: int
        """
        return 1 if np.dot(self.weight, np.array(x)) + self.bias > 0 else 0

    def compute_with_threshold(self, x):
        """
        Вычисление с помощью порогового значения.

        :param x: список входных параметров
        :type x: list
        :return: выход перцептрона
        :rtype: int
        """
        return 1 if np.dot(self.weight, np.array(x)) > -self.bias else 0
