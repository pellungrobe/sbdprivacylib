from privlib.anonymization.src.algorithms.algorithm import Algorithm
from privlib.anonymization.src.entities.dataset import Dataset
from tqdm.auto import tqdm
import copy


class Microaggregation(Algorithm):
    """Microaggregation

    Class that implements the microaggregation clustering algorithm.
    This algorithm performs a clustering of records faster than MDAV but
    less accurate in terms of information loss.
    This algorithm implementation can be executed by the anonymization scheme due to its extends
    Algorithm class and implements the necessary methods.
    (See examples of use in section 3 of the jupyter notebook: test_anonymization.ipynb)
    (See also the files "test_k_anonymity.py" and "test_differential_privacy" in the folder "tests")

    See Also
    --------
    :class:`Algorithm`

    References
    ----------
    References
    ----------
    .. [2] J.Soria-Comas, J.Domingo-Ferrer, D.Sánchez and S.Martínez, "t-Closeness through microaggregation: strict privacy with enhanced utility preservation", IEEE Transactions on Knowledge and Data Engineering, Vol. 27, no. 11, pp. 3098-3110, Oct 2015. DOI: https://doi.org/10.1109/TKDE.2015.2435777

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
            The desired level of clusters (size of cluster >= k).
        Returns
        -------
        :list of list of :class:`Record`
            A list where each item is a list a cluster of records.

        See Also
        --------
            class:`Record`
        """
        records = copy.deepcopy(records)
        pbar = tqdm(total=len(records))
        Dataset.calculate_standard_deviations(records)
        records.sort()
        num_rec = len(records)
        clusters = []
        remain = num_rec
        # Creating clusters of size k
        while remain >= (2 * k):
            cluster = records[:k]
            clusters.append(cluster)
            records = records[k:]
            remain -= k
            pbar.update(k)
        # Remaining values in a cluster
        clusters.append(records)
        pbar.update(len(records))
        pbar.close()

        return clusters

    def __str__(self):
        return "Microaggregation"
