from privlib.anonymization.src.entities.record import Record
from abc import ABC, abstractmethod


class Algorithm(ABC):
    """Algorithm

    Abstract class that represents a clustering algorithm.
    Defines a series of functions necessaries in all clustering algorithms.
    Classes implementing a clustering algorithm must extend this class.

    """

    @staticmethod
    @abstractmethod
    def create_clusters(records, k):
        """create_clusters

        Function to perform the clustering of the records given as parameter
        Abstract method, all clustering algorithms must implement it.

        Parameters
        ----------
        records : list of :class:`Record`
            the list of records to perform the clustering.

        k : integer
            The minimum number of records in each cluster
        Returns
        -------
        list
             return a list where each item is a list of Record corresponding to a cluster of size >= k.

        See Also
        --------
        :class:`Record`
        """
        pass

    @staticmethod
    def calculate_centroid(records, **kwargs):
        """calculate_centroid

        Function that calculates the centroid of a list of records.
        The centroid is formed as the centroid of each attribute.
        Each attribute type value implements its centroid calculation (see :class:`Value`)

        Parameters
        ----------
        records : list of Record
            the list of records to calculate the centroid.

        **kwargs : optional
            Additional arguments that the specific attribute type value may need to calculate the centroid
        Returns
        -------
        :class:`Record`
            A record that is the centroid of the list of records
        See Also
        --------
        :class:`Record`
        :class:`Value`
        """
        centroid_values = []
        for i in range(len(records[0].values)):
            # treat only quasi-identifiers
            if Record.reference_record.values[i] is not None:
                attr_data = []
                for j in range(len(records)):
                    attr_data.append(records[j].values[i])
                centroid = attr_data[0].calculate_centroid(attr_data, **kwargs)
                centroid_values.append(centroid)
            else:
                centroid_values.append("")
        centroid = Record(0, centroid_values)

        return centroid

    @abstractmethod
    def __str__(self):
        pass
