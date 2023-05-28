from privlib.anonymization.src.attribute_types.value import Value
from privlib.anonymization.src.utils import constants
from privlib.anonymization.src.utils import utils
from datetime import datetime
import numpy as np
import pandas as pd


class Datetime(Value):
    """Datetime

    Class that implements the necessary methods to deal with attribute type datetime values

    """

    def __init__(self, value):
        """Constructor, called from inherited classes
        Creates an instance of the attribute type for datetime values

        Parameters
        ----------
        value :
            the date is received as a string: yyyy-mm-dd hh:mm:ss

        See Also
        --------
        :class:`Value`
        """
        self.value = value
        value_datetime = None
        if type(value) is str:
            value_datetime = datetime.fromisoformat(value)
        elif type(value) is pd.Timestamp:
            value_datetime = value.to_pydatetime()
        self.id = 0
        self.timestamp = value_datetime.timestamp()

    def distance(self, value):
        """distance

        Calculates the distance between this datetime and the received value

        Parameters
        ----------
        value :
            The other datetime to calculate the distance.

        Returns
        -------
        float
            The distance between the two datetimes.

        See Also
        --------
        :class:`Value`
        """
        return self.timestamp - value.timestamp

    @staticmethod
    def calculate_centroid(values, **kwargs):
        """calculate_centroid

        Calculates the datetime that is the centroid of the list of datetimes given as parameter
        The centroid is the mean of datetimes

        Parameters
        ----------
        values :
            The list of datetimes to calculate its centroid

        **kwargs : optional
            Additional arguments that the specific attribute type value may need to calculate the centroid

        Returns
        -------
        Datetime
            The datetime that is the centroid of the list of datetimes.
        """
        if constants.EPSILON in kwargs.keys():
            centroid = Datetime.calculate_dp_centroid(values, **kwargs)
            return centroid
        # date resulting of the mean of timestamps
        mean = Datetime.calculate_mean(values)
        centroid = Datetime(mean)

        return centroid

    @staticmethod
    def calculate_dp_centroid(values, **kwargs):
        """calculate_centroid

        Calculates the datetime that is the differential private centroid of the list datetimes given as parameter
        The centroid is the mean of datetimes with a laplace noise added and bounded to the
        min and max possible values.

        Parameters
        ----------
        values :
            The list of datetimes to calculate its centroid

        **kwargs : optional
            Additional arguments that the specific attribute type value may need to calculate the centroid

        Returns
        -------
        Datetime
            The datetime that is the differential private centroid of the list of datetimes.
        """
        mean = Datetime.calculate_mean(values)
        mean = Datetime.datetime_to_timestamp(mean)
        epsilon = float(kwargs[constants.EPSILON])
        k = float(kwargs[constants.K])
        max_value = Datetime.datetime_to_timestamp(kwargs[constants.MAX_VALUE])
        min_value = Datetime.datetime_to_timestamp(kwargs[constants.MIN_VALUE])
        scale = (max_value - min_value) / (k * epsilon)
        dp_centroid = utils.add_laplace_noise(mean, scale, max_value, min_value)
        dp_centroid = Datetime.timestamp_to_datetime(dp_centroid)
        dp_centroid = Datetime(dp_centroid)

        return dp_centroid

    @staticmethod
    def sort(values):
        """sort

        Sorts the list of datetimes received as parameter. The list is sorted in function of
        distance of each element to the datetime reference value.

        Parameters
        ----------
        values :
            The list of datetime to calculate its centroid
        """
        values.sort(key=lambda x: x.timestamp)

    @staticmethod
    def calculate_standard_deviation(values):
        """calculate_standard_deviation

        Calculates the standard deviation of the list of datetimes received as parameter.

        Parameters
        ----------
        values :
            The list of datetimes to calculate the standard deviation

        Returns
        -------
        float
            The standard deviation of the list of datetimes.
        """
        values_temp = []
        for value in values:
            values_temp.append(value.timestamp)

        return np.std(values_temp)

    @staticmethod
    def calculate_mean(values):
        """calculate_mean

        Calculates the mean of the list of datetimes received as parameter.
        The mean consist of the mean of the timestamps resulting of the datetimes
        Parameters
        ----------
        values :
            The list of datetimes to calculate the mean

        Returns
        -------
        Datetime
            The mean of the list of datetimes.
        """
        mean = 0
        for value in values:
            mean += value.timestamp
        mean /= len(values)
        mean = Datetime.timestamp_to_datetime(mean)

        return mean

    @staticmethod
    def calculate_variance(values):
        """calculate_variance

        Calculates the variance of the list of datetimes received as parameter.

        Parameters
        ----------
        values :
            The list of datetimes to calculate the variance

        Returns
        -------
        float
            The variance of the list of datetimes.
        """
        mean = Datetime.calculate_mean(values)
        mean = Datetime(mean)
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

        Calculates the min and max value of the list of datetimes received as parameter.

        Parameters
        ----------
        values :
            The list of datetimes to calculate the min and max value

        Returns
        -------
        Date, Date
            The min and max datetimes.
        """
        mini = min(values, key=lambda x: x.timestamp)
        maxi = max(values, key=lambda x: x.timestamp)
        maxi_margin = maxi.timestamp * margin
        mini = mini.timestamp - (maxi_margin - maxi.timestamp)
        maxi = maxi_margin
        d = datetime.fromtimestamp(mini)
        mini = str(d)
        d = datetime.fromtimestamp(maxi)
        maxi = str(d)

        return mini, maxi

    @staticmethod
    def calculate_reference_value(values):
        """calculate_reference_value

        Calculates the reference value of the list of datetimes received as parameter.

        Parameters
        ----------
        values :
            The list of datetimes to calculate the reference value

        Returns
        -------
        Datetime
            The datetime reference value.
        """
        Datetime.reference_value = min(values, key=lambda x: x.timestamp)

        return Datetime.reference_value

    @staticmethod
    def datetime_to_timestamp(date):
        date = datetime.fromisoformat(date)
        return date.timestamp()

    @staticmethod
    def timestamp_to_datetime(timestamp):
        d = datetime.fromtimestamp(timestamp)
        return str(d)

    def __eq__(self, other):
        return self.timestamp == other.timestamp

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __str__(self):
        return str(self.value)
