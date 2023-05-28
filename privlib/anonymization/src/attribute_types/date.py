from privlib.anonymization.src.attribute_types.value import Value
from privlib.anonymization.src.utils import constants
from privlib.anonymization.src.utils import utils
from datetime import datetime
import numpy as np


class Date(Value):
    """Date

    Class that implements the necessary methods to deal with attribute type date values

    """

    def __init__(self, value):
        """Constructor, called from inherited classes
        Creates an instance of the attribute type for date values

        Parameters
        ----------
        value :
            the date is received as a string: dd/mm/yyyy

        See Also
        --------
        :class:`Value`
        """
        self.value = value
        self.id = 0
        self.timestamp = Date.date_to_timestamp(self.value)

    def distance(self, value):
        """distance

        Calculates the distance between this date and the received value

        Parameters
        ----------
        value :
            The other date to calculate the distance.

        Returns
        -------
        float
            The distance between the two dates.

        See Also
        --------
        :class:`Value`
        """
        return self.timestamp - value.timestamp

    @staticmethod
    def calculate_centroid(values, **kwargs):
        """calculate_centroid

        Calculates the date that is the centroid of the list of dates given as parameter
        The centroid is the mean of dates

        Parameters
        ----------
        values :
            The list of dates to calculate its centroid

        **kwargs : optional
            Additional arguments that the specific attribute type value may need to calculate the centroid

        Returns
        -------
        Date
            The date that is the centroid of the list of dates.
        """
        if constants.EPSILON in kwargs.keys():
            centroid = Date.calculate_dp_centroid(values, **kwargs)
            return centroid
        # date resulting of the mean of timestamps
        mean = Date.calculate_mean(values)
        centroid = Date(mean)

        return centroid

    @staticmethod
    def calculate_dp_centroid(values, **kwargs):
        """calculate_centroid

        Calculates the date that is the differential private centroid of the list dates given as parameter
        The centroid is the mean of dates with a laplace noise added and bounded to the
        min and max possible values.

        Parameters
        ----------
        values :
            The list of dates to calculate its centroid

        **kwargs : optional
            Additional arguments that the specific attribute type value may need to calculate the centroid

        Returns
        -------
        Date
            The date that is the differential private centroid of the list of dates.
        """
        mean = Date.calculate_mean(values)
        mean = Date.date_to_timestamp(mean)
        epsilon = float(kwargs[constants.EPSILON])
        k = float(kwargs[constants.K])
        max_value = Date.date_to_timestamp(kwargs[constants.MAX_VALUE])
        min_value = Date.date_to_timestamp(kwargs[constants.MIN_VALUE])
        scale = (max_value - min_value) / (k * epsilon)
        dp_centroid = utils.add_laplace_noise(mean, scale, max_value, min_value)
        dp_centroid = Date.timestamp_to_date(dp_centroid)
        dp_centroid = Date(dp_centroid)

        return dp_centroid

    @staticmethod
    def sort(values):
        """sort

        Sorts the list of dates received as parameter. The list is sorted in function of
        distance of each element to the date reference value.

        Parameters
        ----------
        values :
            The list of date to calculate its centroid
        """
        values.sort(key=lambda x: x.timestamp)

    @staticmethod
    def calculate_standard_deviation(values):
        """calculate_standard_deviation

        Calculates the standard deviation of the list of dates received as parameter.

        Parameters
        ----------
        values :
            The list of dates to calculate the standard deviation

        Returns
        -------
        float
            The standard deviation of the list of dates.
        """
        values_temp = []
        for value in values:
            values_temp.append(value.timestamp)

        return np.std(values_temp)

    @staticmethod
    def calculate_mean(values):
        """calculate_mean

        Calculates the mean of the list of dates received as parameter.
        The mean consist of the mean of the timestamps resulting of the dates
        Parameters
        ----------
        values :
            The list of dates to calculate the mean

        Returns
        -------
        Date
            The mean of the list of dates.
        """
        mean = 0
        for value in values:
            mean += value.timestamp
        mean /= len(values)
        mean = Date.timestamp_to_date(mean)

        return mean

    @staticmethod
    def calculate_variance(values):
        """calculate_variance

        Calculates the variance of the list of dates received as parameter.

        Parameters
        ----------
        values :
            The list of dates to calculate the variance

        Returns
        -------
        float
            The variance of the list of dates.
        """
        mean = Date.calculate_mean(values)
        mean = Date(mean)
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

        Calculates the min and max value of the list of dates received as parameter.

        Parameters
        ----------
        values :
            The list of dates to calculate the min and max value

        Returns
        -------
        Date, Date
            The min and max dates.
        """
        mini = min(values, key=lambda x: x.timestamp)
        maxi = max(values, key=lambda x: x.timestamp)
        maxi_margin = maxi.timestamp * margin
        mini = mini.timestamp - (maxi_margin - maxi.timestamp)
        maxi = maxi_margin
        d = datetime.fromtimestamp(mini)
        mini = str(d.day) + "/" + str(d.month) + "/" + str(d.year)
        d = datetime.fromtimestamp(maxi)
        maxi = str(d.day) + "/" + str(d.month) + "/" + str(d.year)

        return mini, maxi

    @staticmethod
    def calculate_reference_value(values):
        """calculate_reference_value

        Calculates the reference value of the list of dates received as parameter.

        Parameters
        ----------
        values :
            The list of dates to calculate the reference value

        Returns
        -------
        Date
            The date reference value.
        """
        Date.reference_value = min(values, key=lambda x: x.timestamp)

        return Date.reference_value

    @staticmethod
    def date_to_timestamp(date):
        date_temp = date.split("/")
        d = datetime(int(date_temp[2]), int(date_temp[1]), int(date_temp[0]))
        return d.timestamp()

    @staticmethod
    def timestamp_to_date(timestamp):
        d = datetime.fromtimestamp(timestamp)
        return str(d.day) + "/" + str(d.month) + "/" + str(d.year)

    def __eq__(self, other):
        return self.timestamp == other.timestamp

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __str__(self):
        return self.value
