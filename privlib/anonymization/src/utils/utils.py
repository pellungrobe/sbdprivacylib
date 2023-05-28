import numpy as np
import pandas as pd
import random
import math
from bisect import bisect_left
from datetime import timedelta


def get_class(fully_qualified_path, module_name, class_name, *instantiation):
    """
    Returns an instantiated class for the given string descriptors
    :param fully_qualified_path: The path to the module eg("anonymization.src.attribute_types.numerical_discrete")
    :param module_name: The module name eg("numerical_discrete")
    :param class_name: The class name eg("Numerical_discrete")
    :param instantiation: Any fields required to instantiate the class eg the value
    :return: An instance of the class
    """
    p = __import__(fully_qualified_path)
    modules = fully_qualified_path.split(".")
    m = getattr(p, modules[1])
    for i in range(2, len(modules)):
        m = getattr(m, modules[i])
    c = getattr(m, class_name)
    instance = c(*instantiation)
    return instance


def random_laplace(mu, scale):
    """
    Returns a random laplace distributed value with location and scale
    :param (float) mu: the position of the distribution peak
    :param (float) scale: The exponential decay
    :return: the laplace random distributed value
    :rtype: float
    """
    return np.random.default_rng().laplace(mu, scale)


def random_laplace2(mu, scale):
    """
    Returns a random laplace distributed value with location and scale
    :param (float) mu: the position of the distribution peak
    :param (float) scale: The exponential decay
    :return: the laplace random distributed value
    :rtype: float
    """
    x = random.random()
    value = inverse_laplace(mu, scale, x)

    return value


def inverse_laplace(mu, scale, x):
    value = x - 0.5
    logarithm = math.log((1 - 2 * (abs(value))))
    sign = float(np.sign(value))
    result = mu - scale * sign * logarithm

    return result


def add_laplace_noise(value, scale, max_value, min_value):
    """
    Add Laplace noise to a value given as parameter.
    The value will be bounded between max and min values given as parameters
    :param (float) value: the value to convert
    :param (float) scale: The exponential decay
    :param (float) max_value: max possible value in the domain
    :param (float) min_value: min possible value in the domain
    :return: The value with the noise added
    :rtype: float
    """
    noise = random_laplace(0, scale)
    dp_value = value + noise
    if dp_value > max_value:
        dp_value = max_value
    if dp_value < min_value:
        dp_value = min_value

    return dp_value


def read_dataframe_from_csv(path_csv):
    df = pd.read_csv(path_csv)
    df.name = path_csv

    return df


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns the index of the closest value to myNumber.
    If two numbers are equally close, return the index of the smallest number.
    :param (list) myList: the list of values
    :param (float) myNumber: The number to be searched
    :return: The index of myList of the closest value to myNumber
    :rtype: int
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return pos
        # return myList[0]
    if pos == len(myList):
        return pos
        # return m/yList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        # return after
        return pos
    else:
        # return before
        return pos - 1


def take_closest_window(myList, myNumber, window_size):
    """
    Assumes myList is sorted. Returns a list of positions of my list with size window_size.
    The window_size positions of the closest values in mylist to myNumber.
    :param (list) myList: the list of values
    :param (float) myNumber: The number to be searched
    :param (float) window_size: The number of positions to be returned
    :return: The index of myList of the closest value to myNumber
    :rtype: list of int
    """
    pos = take_closest(myList, myNumber)
    cut = int(window_size / 2)
    rest_before = 0
    rest_after = 0
    pos_before = pos - cut
    if window_size % 2 == 0:
        pos_before += 1
    if pos_before < 0:
        rest_before = pos_before * -1
        pos_before = 0
    pos_after = pos + cut
    if pos_after > len(myList) - 1:
        rest_after = pos_after - (len(myList) - 1)
        pos_after = len(myList) - 1
    pos_before -= rest_after
    pos_after += rest_before

    return [x for x in range(pos_before, pos_after + 1)]


def format_time(seconds):
    runtime = str(timedelta(seconds=seconds)).split(".")[0]
    return runtime
