from privlib.anonymization.src.algorithms.algorithm import Algorithm
from privlib.anonymization.src.entities.dataset import Dataset
from tqdm.auto import tqdm
import numpy as np


class Mdav(Algorithm):
    """MDAV

    Class that implements the MDAV clustering algorithm.
    The MDAV algorithm performs an accurate clustering of records being the computational cost is quadratic
    This algorithm implementation can be executed by the anonymization scheme due to its extends
    Algorithm class  and implements the necessary methods.
    (See examples of use in sections 2 of the jupyter notebook: test_anonymization.ipynb)
    (See also the files "test_k_anonymity.py" and "test_differential_privacy" in the folder "tests")

    See Also
    --------
    :class:`Algorithm`

    References
    ----------
    .. [1] Josep Domingo-Ferrer and Vicenç Torra, "Ordinal, continuous and heterogeneous k-anonymity through microaggregation", Data Mining and Knowledge Discovery, Vol. 11, pp. 195-212, Sep 2005. DOI: https://doi.org/10.1007/s10618-005-0007-5

    """

    @staticmethod
    def create_clusters(records, k):
        """create_clusters

        Function to perform the clustering of the list of records given as parameter.
        The size of the resulting clusters will be >= k

        Parameters
        ----------
        records : list of :class:`Record`
            The list of records to perform the clustering.

        k : int
            The desired level of clusters (size of cluster >= k)
        Returns
        -------
        : :list of list of :class:`Record`
            A list where each item is a list a cluster of records.

        See Also
        --------
        :class:`Record`
        """
        pbar = tqdm(total=len(records))
        D = np.array(records)
        Dataset.calculate_standard_deviations(D)
        clusters = []
        while len(D) >= 3 * k:
            centroid = Mdav.calculate_centroid(D)
            # calculate r (furthest from centroid)
            r, i = Mdav.calculate_furthest(centroid, D)
            D = np.delete(D, i)
            D, cluster = Mdav.create_cluster(D, r, k)
            clusters.append(cluster)
            pbar.update(k)
            # calculate s (Furthest from r)
            s, i = Mdav.calculate_furthest(r, D)
            D = np.delete(D, i)
            D, cluster = Mdav.create_cluster(D, s, k)
            clusters.append(cluster)
            pbar.update(k)
        if len(D) >= 2 * k:
            centroid = Mdav.calculate_centroid(D)
            # calculate r (furthest from centroid)
            r, i = Mdav.calculate_furthest(centroid, D)
            D = np.delete(D, i)
            D, cluster = Mdav.create_cluster(D, r, k)
            clusters.append(cluster)
            pbar.update(k)
        if len(D) > 0:
            cluster = list(D)
            clusters.append(cluster)
            pbar.update(len(cluster))
            pbar.close()

        return clusters

    @staticmethod
    def distance(c1, c2):
        return c1.distance(c2)

    @staticmethod
    def calculate_furthest(record, records):
        distances = [Mdav.distance(v, record) for v in records]
        index = np.argmax(distances)
        furthest = records[index]
        return furthest, index

    @staticmethod
    def create_cluster(records, record, k):
        distances = [Mdav.distance(v, record) for v in records]
        records = records[np.argpartition(distances, k - 1)]
        c = [record]
        c.extend(records[: k - 1])
        records = records[k - 1 :]
        return records, c

    def __str__(self):
        return "MDAV"
