from privlib.anonymization.src.attribute_types.value import Value
from privlib.anonymization.src.utils import constants
from privlib.anonymization.src.utils import utils
import numpy as np


class Numerical_discrete(Value):
    """Numerical_discrete

    Class that implements the necessary methods to deal with attribute type numerical discrete values

    """

    def __init__(self, value):
        """Constructor, called from inherited classes
        Creates an instance of the attribute type for numerical discrete values

        Parameters
        ----------
        value :
            the numerical discrete is received as string representing an integer value

        See Also
        --------
        :class:`Value`
        """
        self.value = int(value)
        self.id = 0

    def distance(self, value):
        """distance

        Calculates the distance between this numerical discrete and the received value

        Parameters
        ----------
        value :
            The other numerical discrete to calculate the distance.

        Returns
        -------
        float
            The distance between the two numerical discrete values.

        See Also
        --------
        :class:`Value`
        """
        return self.value - value.value

    @staticmethod
    def calculate_centroid(values, **kwargs):
        """calculate_centroid

        Calculates the numerical discrete that is the centroid of the list of numerical discrete values
        given as parameter. The centroid is the mean of numerical discrete values

        Parameters
        ----------
        values :
            The list of numerical discrete values to calculate its centroid

        **kwargs : optional
            Additional arguments that the specific attribute type value may need to calculate the centroid

        Returns
        -------
        Numerical_discrete
            The value that is the centroid of the list of numerical discrete values.
        """
        if constants.EPSILON in kwargs.keys():
            centroid = Numerical_discrete.calculate_dp_centroid(values, **kwargs)
            return centroid
        # mean
        mean = Numerical_discrete.calculate_mean(values)
        centroid = Numerical_discrete(str(round(mean)))

        return centroid

    @staticmethod
    def calculate_dp_centroid(values, **kwargs):
        """calculate_centroid

        Calculates the numerical discrete value that is the differential private centroid of
        the list numerical discrete values given as parameter
        The centroid is the mean of numerical discrete values with a laplace noise added and bounded to the
        min and max possible values.

        Parameters
        ----------
        values :
            The list of numerical discrete values to calculate its centroid

        **kwargs : optional
            Additional arguments that the specific attribute type value may need to calculate the centroid

        Returns
        -------
        Numerical_discrete
            The value that is the differential private centroid of the list of numerical discrete values.
        """
        mean = Numerical_discrete.calculate_mean(values)
        epsilon = float(kwargs[constants.EPSILON])
        k = float(kwargs[constants.K])
        max_value = int(np.rint(float(kwargs[constants.MAX_VALUE])))
        min_value = int(np.rint(float(kwargs[constants.MIN_VALUE])))
        scale = (max_value - min_value) / (k * epsilon)
        dp_centroid = utils.add_laplace_noise(mean, scale, max_value, min_value)
        dp_centroid = Numerical_discrete(str(round(dp_centroid)))

        return dp_centroid

    @staticmethod
    def sort(values):
        """sort

        Sorts the list of numerical discrete values received as parameter. The list is sorted in function of
        distance of each element to the numerical discrete reference value.

        Parameters
        ----------
        values :
            The list of numerical discrete values to calculate its centroid
        """
        values.sort(key=lambda x: x.value)

    @staticmethod
    def calculate_standard_deviation(values):
        """calculate_standard_deviation

        Calculates the standard deviation of the list of numerical discrete values received as parameter.

        Parameters
        ----------
        values :
            The list of numerical discrete values to calculate the standard deviation

        Returns
        -------
        float
            The standard deviation of the list of numerical discrete values.
        """
        values_temp = []
        for value in values:
            values_temp.append(value.value)

        return np.std(values_temp)

    @staticmethod
    def calculate_mean(values):
        """calculate_mean

        Calculates the mean of the list of numerical discrete values received as parameter.
        The mean consist of the mean of the numerical discrete values
        Parameters
        ----------
        values :
            The list of numerical discrete values to calculate the mean

        Returns
        -------
        float
            The mean of the list of numerical discrete values.
        """
        mean = 0
        for value in values:
            mean += value.value
        mean /= len(values)

        return mean

    @staticmethod
    def calculate_variance(values):
        """calculate_variance

        Calculates the variance of the list of numerical discrete values received as parameter.

        Parameters
        ----------
        values :
            The list of numerical discrete values to calculate the variance

        Returns
        -------
        float
            The variance of the list of numerical discrete values.
        """
        mean = Numerical_discrete.calculate_mean(values)
        mean = Numerical_discrete(str(round(mean)))
        variance = 0
        for value in values:
            partial = value.distance(mean)
            partial = partial * partial
            variance += partial
        variance /= len(values)

        return variance

    @staticmethod
    def calculate_min_max(values, margin):
        """calculate_min_max

        Calculates the min and max values of the list of numerical discrete values received as parameter.

        Parameters
        ----------
        values :
            The list of numerical discrete values to calculate the min and max values

        Returns
        -------
        int, int
            The min and max numerical discrete values.
        """
        mini = min(values)
        maxi = max(values)
        maxi_margin = maxi * margin
        mini -= maxi_margin - maxi
        maxi = maxi_margin

        return mini, maxi

    @staticmethod
    def calculate_reference_value(values):
        """calculate_reference_value

        Calculates the reference value of the list of numerical discrete values received as parameter.

        Parameters
        ----------
        values :
            The list of numerical discrete values to calculate the reference value

        Returns
        -------
        Numerical_discrete
            The numerical discrete reference value.
        """
        Numerical_discrete.reference_value = min(values)

        return Numerical_discrete.reference_value

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __str__(self):
        return str(self.value)
