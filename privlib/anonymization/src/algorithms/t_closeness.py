from privlib.anonymization.src.algorithms.anonymization_scheme import (
    Anonymization_scheme,
)
from privlib.anonymization.src.utils.sensitivity_type import Sensitivity_type
from privlib.anonymization.src.utils import utils
import copy
from timeit import default_timer as timer


class T_closeness(Anonymization_scheme):
    """T_closeness

    Class that implements k-t-closeness anonymization method.
    This algorithm implementation can be executed by the anonymization scheme due to its extends
    Anonymization_scheme class
    (See examples of use in section 4 of the jupyter notebook: test_anonymization.ipynb)
    (See also the file "test_t_closeness.py" in the folder "tests")

    See Also
    --------
    :class:`Anonymization_scheme`

    References
    ----------
    .. [2] Jordi Soria-Comas, Josep Domingo-Ferrer, David Sánchez and Sergio Martínez, "t-Closeness through microaggregation: strict privacy with enhanced utility preservation", IEEE Transactions on Knowledge and Data Engineering, Vol. 27, no. 11, pp. 3098-3110, Oct 2015. DOI: https://doi.org/10.1109/TKDE.2015.2435777

    """

    def __init__(self, original_dataset, k, t):
        """Constructor, called from inherited classes

        Parameters
        ----------
        original_dataset : :class:`Dataset`
            the data set to be anonymized.

        k : int
            The size of the clusters in the clustering process

        t : float
            The desired level of t-closeness privacy in the confidential attribute

        See Also
        --------
        :class:`Dataset`
        """
        super().__init__(original_dataset)
        self.k = k
        self.t = t

    def calculate_anonymization(self, algorithm):
        """calculate_anonymization

        Function to perform the k-t-closeness anonymization method.

        Parameters
        ----------
        algorithm : :class:`Algorithm`
            The clustering algorithm used during the anonymization.

        See Also
        --------
        :class:`Algorithm`
        """
        t_ini = timer()
        print("Anonymizing " + str(self) + " via " + str(algorithm))
        clusters = self.create_k_t_clusters()
        self.anonymized_dataset = copy.deepcopy(self.original_dataset)
        print("Anonymizing")
        for cluster in clusters:
            centroid = algorithm.calculate_centroid(cluster)
            for record in cluster:
                for i in range(self.original_dataset.num_attr):
                    name = self.original_dataset.header[i]
                    attribute = self.original_dataset.attributes[name]
                    sensitivity = attribute.sensitivity_type
                    if sensitivity != Sensitivity_type.QUASI_IDENTIFIER.value:
                        continue
                    self.anonymized_dataset.records[record.id].values[
                        i
                    ] = centroid.values[i]
        self.suppress_identifiers()
        Anonymization_scheme.runtime = timer() - t_ini
        print(
            f"Anonymization runtime: {utils.format_time(Anonymization_scheme.runtime)}"
        )

    def create_k_t_clusters(self):
        self.anonymized_dataset = copy.deepcopy(self.original_dataset)
        index_confidential = self.get_index_confidential_attribute()
        print(
            "Sorting by confidential attribute: "
            + self.original_dataset.header[index_confidential]
        )
        conf_attr_values = []
        for record in self.anonymized_dataset.records:
            value = record.values[index_confidential]
            value.id = record.id
            conf_attr_values.append(value)
        conf_attr_values[0].sort(conf_attr_values)
        records_temp = []
        for value in conf_attr_values:
            records_temp.append(self.anonymized_dataset.records[value.id])
        for i in range(len(records_temp)):
            self.anonymized_dataset.records[i] = records_temp[i]

        n = len(self.anonymized_dataset)
        k_prime = n / (2 * (n - 1) * self.t + 1)
        if self.k > k_prime:
            num_clusters_k = self.k
        else:
            num_clusters_k = int(k_prime) + 1
        num_item = int(len(self.anonymized_dataset) / num_clusters_k)
        remainder = len(self.anonymized_dataset) % num_clusters_k
        if remainder >= num_item:
            num_clusters_k = num_clusters_k + (remainder / num_item)
        num_clusters_k = int(num_clusters_k)
        print("Creating k subsets (" + str(num_clusters_k) + ")")
        clusters_k = []
        index = 0
        for i in range(num_clusters_k):
            cluster = []
            for j in range(num_item):
                r = self.anonymized_dataset.records[index]
                cluster.append(r)
                index += 1
            clusters_k.append(cluster)
        # remain records in a subset
        if index < len(self.anonymized_dataset):
            cluster = []
            for i in range(index, len(self.anonymized_dataset)):
                r = self.anonymized_dataset.records[index]
                cluster.append(r)
            clusters_k.append(cluster)

        print("Sorting each subset by quasi-identifiers")
        for cluster in clusters_k:
            cluster.sort()

        print("Creating clusters")
        remain = len(self.anonymized_dataset)
        clusters = []
        index = 0
        while remain > 0:
            cluster = []
            for cluster_k in clusters_k:
                if len(cluster_k) > index:
                    cluster.append(cluster_k[index])
                    remain -= 1
            index += 1
            clusters.append(cluster)

        return clusters

    def get_index_confidential_attribute(self):
        for index, attribute_name in enumerate(self.original_dataset.header):
            attribute = self.original_dataset.attributes[attribute_name]
            if attribute.sensitivity_type == Sensitivity_type.CONFIDENTIAL.value:
                return index

    def __str__(self):
        return "k-t-Closeness, k = " + str(self.k) + ", t = " + str(self.t)
