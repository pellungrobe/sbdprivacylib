from anonymization.src.attribute_types.value import Value
from anonymization.src.utils import constants
from anonymization.src.utils import utils
import numpy as np
from collections import Counter


class Plain_categorical(Value):
    """Plain_categorical

    Class that implements the necessary methods to deal with attribute type Plain categorical

    """
    reference_value = None
    rank_values = {}

    def __init__(self, value):
        """Constructor, called from inherited classes
        Creates an instance of the attribute type for plain categorical value

        Parameters
        ----------
        value :
            the plain categorical is received as an string

        See Also
        --------
        :class:`Value`
        """
        self.value = value
        self.id = 0
        self.rank = 0
        self.hash = hash(value)

    def distance(self, value):
        """distance

        Calculates the distance between this plain categorical and the received value

        Parameters
        ----------
        value :
            The other plain categorical value to calculate the distance.

        Returns
        -------
        float
            The distance between the two plain categorical values.

        See Also
        --------
        :class:`Value`
        """
        if self.value == value.value:
            return 0.0
        else:
            return 1.0

    @staticmethod
    def calculate_centroid(values, **kwargs):
        """calculate_centroid

        Calculates the plain categorical value that is the centroid of the list of plain categorical values
        given as parameter. The centroid is the most common value (the mode) of the list of plain categorical values

        Parameters
        ----------
        values :
            The list of plain categorical values to calculate its centroid

        **kwargs : optional
            Additional arguments that the specific attribute type value may need to calculate the centroid

        Returns
        -------
        Plain categorical
            The value that is the centroid of the list of plain categorical values.
        """
        if constants.EPSILON in kwargs.keys():
            centroid = Plain_categorical.calculate_dp_centroid(values, **kwargs)
            return centroid
        # mode
        centroid = Counter(values).most_common()[0][0]

        return Plain_categorical(centroid.value)

    @staticmethod
    def calculate_dp_centroid(values, **kwargs):
        """calculate_centroid

        Calculates the plain categorical value that is the differential private centroid of
        the list of plain categorical values given as parameter
        The centroid is the value labeled as the mean of labels with a laplace noise added to the mean label and bounded to the
        min and max possible values.

        Parameters
        ----------
        values :
            The list of plain categorical values to calculate its centroid

        **kwargs : optional
            Additional arguments that the specific attribute type value may need to calculate the centroid

        Returns
        -------
        Plain categorical
            The value that is the differential private centroid of the list of plain categorical values.
        """
        # dp noise applied on the index of values
        epsilon = float(kwargs[constants.EPSILON])
        k = float(kwargs[constants.K])
        max_value = len(Plain_categorical.rank_values)
        min_value = 1
        scale = (max_value - min_value) / (k * epsilon)
        rankings = [value.rank for value in values]
        mean_ranking = sum(rankings) / len(rankings)
        dp_rank = utils.add_laplace_noise(mean_ranking, scale, max_value, min_value)
        dp_rank = np.rint(dp_rank)
        centroid = Plain_categorical.rank_values[dp_rank]

        return Plain_categorical(centroid)

    @staticmethod
    def sort(values):
        """sort

        Sorts the list of plain categorical values received as parameter. The list is sorted in function of
        the frequency of each plain categorical value in the list of values.

        Parameters
        ----------
        values :
            The list of plain categorical values to calculate its centroid
        """
        counts = Counter(values)
        values_ord = sorted(values, key=lambda x: -counts[x])
        for i, value in enumerate(values_ord):
            values[i] = value
        c = Counter(values).most_common()
        control = {}
        for rank, item in enumerate(c):
            value = item[0]
            control[value] = rank+1
            Plain_categorical.rank_values[rank+1] = value.value
        for value in values:
            value.rank = control[value]

    @staticmethod
    def calculate_standard_deviation(values):
        """calculate_standard_deviation

        Calculates the standard deviation of the list of plain categorical values received as parameter.

        Parameters
        ----------
        values :
            The list of plain categorical values to calculate the standard deviation

        Returns
        -------
        Plain_categorical
            The standard deviation of the list of plain categorical values, in this case 0.5.
        """
        # we need to have calculated rank_values
        Plain_categorical.rank_values = {}
        Plain_categorical.sort(values)
        return 0.5

    @staticmethod
    def calculate_mean(values):
        """calculate_mean

        Calculates the mean of the list of plain categorical values received as parameter.
        The mean consist of the most common value of the plain categorical values
        Parameters
        ----------
        values :
            The list of plain categorical values to calculate the mean

        Returns
        -------
        Plain_categorical
            The mean of the list of plain categorical values.
        """
        mean = Counter(values).most_common()[0][0]

        return mean

    @staticmethod
    def calculate_variance(values):
        """calculate_variance

        Calculates the variance of the list of plain categorical values received as parameter.
        The variance is calculated in function of the distance to the mean

        Parameters
        ----------
        values :
            The list of plain categorical values to calculate the variance

        Returns
        -------
        float
            The variance of the list of plain categorical values.
        """
        mean = Plain_categorical.calculate_mean(values)
        mean = Plain_categorical(mean.value)
        variance = 0
        for value in values:
            partial = value.distance(mean)
            partial = partial * partial
            variance += partial
        variance /= len(values)

        return variance

    @staticmethod
    def calculate_min_max(self, margin):
        """calculate_min_max

        Calculates the min and max value of the list of plain categorical values received as parameter.
        In this case, there are not max and min values.

        Parameters
        ----------
        values :
            The list of plain categorical values to calculate the min and max value

        Returns
        -------
        int, int
            The min and max plain categorical values.
        """
        return None, None

    @staticmethod
    def calculate_reference_value(values):
        """calculate_reference_value

        Calculates the reference value of the list of plain categorical values received as parameter.
        For plain categorical, we take the most common value as reference value.

        Parameters
        ----------
        values :
            The list of plain categorical values to calculate the reference value

        Returns
        -------
        Plain_categorical
            The plain categorical reference value.
        """
        Plain_categorical.reference_value = Counter(values).most_common()[0][0]

        return Plain_categorical.reference_value

    def __eq__(self, value):
        return self.value == value.value

    def __lt__(self, other):
        return self.distance(Plain_categorical.reference_value) < other.distance(Plain_categorical.reference_value)

    def __cmp__(self, value):
        if self.value < value.value:
            return -1
        elif self.value > value.value:
            return 1
        else:
            return 0

    def __gt__(self, value):
        return self.value > value.value

    def __hash__(self):
        return self.hash

    def __str__(self):
        return str(self.value)
