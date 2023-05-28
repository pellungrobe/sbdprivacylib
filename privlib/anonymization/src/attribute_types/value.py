from abc import ABC, abstractmethod


class Value(ABC):
    """Value

    Class that represents a value in the record. It is the interface to be inherited by
    a supported attribute type. The different type of values should implement these methods
    according the characteristics of the value.

    """
    @abstractmethod
    def distance(self, value):
        """distance

        Calculates the distance between the self and the received values

        Parameters
        ----------
        value :
            The other value to calculate the distance.

        Returns
        -------
        float
            The distance between the two values.
        """
        pass

    @staticmethod
    @abstractmethod
    def calculate_centroid(values, **kwargs):
        """calculate_centroid

        Calculates the value that is the centroid of the list of values
        given as parameter. The centroid is calculated depending of the attribute type,

        Parameters
        ----------
        values :
            The list of values to calculate its centroid

        **kwargs : optional
            Additional arguments that the specific attribute type value may need to calculate the centroid

        Returns
        -------
        Value
            The value that is the centroid of the list of values.
        """
        pass

    @staticmethod
    @abstractmethod
    def calculate_standard_deviation(values):
        """calculate_standard_deviation

        Calculates the standard deviation of the list of values received as parameter.

        Parameters
        ----------
        values :
            The list of numerical discrete values to calculate the standard deviation

        Returns
        -------
        float
            The standard deviation of the list of values.
        """
        pass

    @staticmethod
    @abstractmethod
    def calculate_mean(values):
        """calculate_mean

        Calculates the mean of the list of values received as parameter.
        The mean calculation depends on the specific attribute type implementation
        Parameters
        ----------
        values :
            The list of values to calculate the mean

        Returns
        -------
        float
            The mean of the list of values.
        """
        pass

    @staticmethod
    @abstractmethod
    def calculate_variance(values):
        """calculate_variance

        Calculates the variance of the list of values received as parameter.
        The variance calculation depends on the specific attribute type implementation

        Parameters
        ----------
        values :
            The list of values to calculate the variance

        Returns
        -------
        float
            The variance of the list of  values.
        """
        pass

    @staticmethod
    @abstractmethod
    def sort(values):
        """sort

        Sorts the list of values received as parameter. The list is sorted in function of
        the specific attribute type implementation.

        Parameters
        ----------
        values :
            The list of values to calculate its centroid
        """
        pass

    @staticmethod
    @abstractmethod
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
        pass

    @staticmethod
    @abstractmethod
    def calculate_reference_value(values):
        """calculate_reference_value

        Calculates the reference value of the list of values received as parameter.
        The reference value is calculated depending of the specific attribute type

        Parameters
        ----------
        values :
            The list of values to calculate the reference value

        Returns
        -------
        Value
            The reference value.
        """
        pass
