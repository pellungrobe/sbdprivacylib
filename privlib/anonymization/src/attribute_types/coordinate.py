from privlib.anonymization.src.attribute_types.value import Value
from privlib.anonymization.src.utils import constants
from privlib.anonymization.src.utils import utils
import math


class Coordinate(Value):
    """Coordinate

    Class that implements the necessary methods to deal with attribute type coordinate values

    """

    reference_value = None

    def __init__(self, value):
        """Constructor, called from inherited classes
        Creates an instance of the attribute type for coordinate values

        Parameters
        ----------
        value :
            the coordinate received as a tuple with (lat,lon) or a list [lat,lon] for spf
            or a string with format "lat:lon" for csv

        See Also
        --------
        :class:`Value`
        """
        if type(value) == list or type(value) == tuple:
            self.value = value
            self.coordinate_lat = self.value[0]
            self.coordinate_lon = self.value[1]
        else:
            temp = value.split(":")
            self.value = temp
            self.coordinate_lat = float(self.value[0])
            self.coordinate_lon = float(self.value[1])
        self.distance_to_reference_value = 0
        self.id = 0

    def distance(self, value):
        """distance

        Calculates the distance between this coordinate and the received value

        Parameters
        ----------
        value :
            The other coordinate to calculate the distance.

        Returns
        -------
        float
            The distance between the two coordinates.

        See Also
        --------
        :class:`Value`
        """
        dist = math.sqrt(
            (value.coordinate_lat - self.coordinate_lat) ** 2
            + (value.coordinate_lon - self.coordinate_lon) ** 2
        )
        return dist

    @staticmethod
    def calculate_centroid(values, **kwargs):
        """calculate_centroid

        Calculates the coordinate that is the centroid of the list coordinates given as parameter
        The centroid is the mean of latitudes, longitudes and times

        Parameters
        ----------
        values :
            The list of coordinates to calculate its centroid

        **kwargs : optional
            Additional arguments that the specific attribute type value may need to calculate the centroid

        Returns
        -------
        Coordinate
            The coordinate that is the centroid of the list of coordinates.
        """
        if constants.EPSILON in kwargs.keys():
            centroid = Coordinate.calculate_dp_centroid(values, **kwargs)
            return centroid
        mean = Coordinate.calculate_mean(values)

        return mean

    @staticmethod
    def calculate_dp_centroid(values, **kwargs):
        """calculate_centroid

        Calculates the coordinate that is the differential private centroid of the list coordinates given as parameter
        The centroid is the mean of latitudes, longitudes and times with a laplace noise added and bounded to the
        min and max possible values.

        Parameters
        ----------
        values :
            The list of coordinates to calculate its centroid

        **kwargs : optional
            Additional arguments that the specific attribute type value may need to calculate the centroid

        Returns
        -------
        Coordinate
            The coordinate that is the differential private centroid of the list of coordinates.
        """
        mean = Coordinate.calculate_mean(values)
        epsilon = float(kwargs[constants.EPSILON])
        k = float(kwargs[constants.K])
        max_value = float(kwargs[constants.MAX_VALUE].coordinate_lat)
        min_value = float(kwargs[constants.MIN_VALUE].coordinate_lat)
        scale = (max_value - min_value) / (k * epsilon)
        dp_centroid_lat = utils.add_laplace_noise(
            mean.coordinate_lat, scale, max_value, min_value
        )
        max_value = float(kwargs[constants.MAX_VALUE].coordinate_lon)
        min_value = float(kwargs[constants.MIN_VALUE].coordinate_lon)
        scale = (max_value - min_value) / (k * epsilon)
        dp_centroid_lon = utils.add_laplace_noise(
            mean.coordinate_lon, scale, max_value, min_value
        )
        dp_centroid = Coordinate([dp_centroid_lat, dp_centroid_lon])

        return dp_centroid

    @staticmethod
    def sort(values):
        """sort

        Sorts the list of coordinates received as parameter. The list is sorted in function of
        distance of each element to the coordinate reference value.

        Parameters
        ----------
        values :
            The list of coordinates to calculate its centroid
        """
        for value in values:
            value.distance_to_reference_value = value.distance(
                Coordinate.reference_value
            )
        values.sort(key=lambda x: x.distance_to_reference_value)

    @staticmethod
    def calculate_standard_deviation(values):
        """calculate_standard_deviation

        Calculates the standard deviation of the list of coordinates received as parameter.

        Parameters
        ----------
        values :
            The list of coordinates to calculate the standard deviation

        Returns
        -------
        float
            The standard deviation of the list of coordinates.
        """
        variance = Coordinate.calculate_variance(values)
        std = math.sqrt(variance)

        return std

    @staticmethod
    def calculate_mean(values):
        """calculate_mean

        Calculates the mean of the list of coordinates received as parameter.

        Parameters
        ----------
        values :
            The list of coordinates to calculate the mean

        Returns
        -------
        float
            The mean of the list of coordinates.
        """
        lats = [v.coordinate_lat for v in values]
        lons = [v.coordinate_lon for v in values]
        mean_lat = sum(lats) / len(lats)
        mean_lon = sum(lons) / len(lons)

        return Coordinate([mean_lat, mean_lon])

    @staticmethod
    def calculate_variance(values):
        """calculate_variance

        Calculates the variance of the list of coordinates received as parameter.

        Parameters
        ----------
        values :
            The list of coordinates to calculate the variance

        Returns
        -------
        float
            The variance of the list of coordinates.
        """
        mean = Coordinate.calculate_mean(values)
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

        Calculates the min and max value of the list of coordinates received as parameter.

        Parameters
        ----------
        values :
            The list of coordinates to calculate the min and max value

        Returns
        -------
        Coordinate, Coordinate
            The min and max coordinates.
        """
        mini = Coordinate([-90, -180])
        maxi = Coordinate([90, 180])

        return mini, maxi

    @staticmethod
    def calculate_reference_value(values):
        """calculate_reference_value

        Calculates the reference value of the list of coordinates received as parameter.

        Parameters
        ----------
        values :
            The list of coordinates to calculate the reference value

        Returns
        -------
        Coordinate
            The coordinate reference value.
        """
        Coordinate.reference_value = Coordinate([-90, -180])

        return Coordinate.reference_value

    def __eq__(self, other):
        if (
            self.value.coordinate_lat == other.value.coordinate_lat
            and self.value.coordinate_lon == other.value.coordinate_lon
        ):
            return True
        else:
            return False

    def __lt__(self, other):
        return self.distance(Coordinate.reference_value) < other.distance(
            Coordinate.reference_value
        )

    def __str__(self):
        s = ""
        s += str(self.coordinate_lat)
        s += ":"
        s += str(self.coordinate_lon)

        return s
