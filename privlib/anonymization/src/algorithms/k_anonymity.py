from privlib.anonymization.src.algorithms.anonymization_scheme import (
    Anonymization_scheme,
)
from privlib.anonymization.src.utils.sensitivity_type import Sensitivity_type
from privlib.anonymization.src.utils import utils
import copy
from timeit import default_timer as timer


class K_anonymity(Anonymization_scheme):
    """K_anonymity

    Class that implements k-anonymity anonymization.
    This algorithm implementation can be executed by the anonymization scheme due to its extends
    Anonymization_scheme class
    (See examples of use in sections 2 and 3 of the jupyter notebook: test_anonymization.ipynb)
    (See also the file "test_k_anonymity.py" in the folder "tests")

    See Also
    --------
    :class:`Anonymization_scheme`

    References
    ----------
    .. [1] Josep Domingo-Ferrer and Vicenç Torra, "Ordinal, continuous and heterogeneous k-anonymity through microaggregation", Data Mining and Knowledge Discovery, Vol. 11, pp. 195-212, Sep 2005. DOI: https://doi.org/10.1007/s10618-005-0007-5

    """

    def __init__(self, original_dataset, k):
        """Constructor, called from inherited classes

        Parameters
        ----------
        original_dataset : :class:`Dataset`
            the data set to be anonymized.

        k : int
            The size of the clusters in the clustering process

        See Also
        --------
        :class:`Dataset`
        """
        super().__init__(original_dataset)
        self.k = k

    def calculate_anonymization(self, algorithm):
        """calculate_anonymization

        Function to perform the k-anonymity anonymization.

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
        clusters = algorithm.create_clusters(self.original_dataset.records, self.k)
        self.anonymized_dataset = copy.deepcopy(self.original_dataset)
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

    def __str__(self):
        return "k-Anonymity, k = " + str(self.k)
