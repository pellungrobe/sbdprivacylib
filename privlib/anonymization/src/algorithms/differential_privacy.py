from privlib.anonymization.src.algorithms.anonymization_scheme import (
    Anonymization_scheme,
)
from privlib.anonymization.src.utils.sensitivity_type import Sensitivity_type
from privlib.anonymization.src.utils import constants
from privlib.anonymization.src.entities.dataset import Dataset
from privlib.anonymization.src.entities.record import Record
from privlib.anonymization.src.utils import utils
import copy
from timeit import default_timer as timer


class Differential_privacy(Anonymization_scheme):
    """Differential_privacy

    Class that implements differential privacy via individual ranking microaggregation-based perturbatuion.
    This algorithm implementation can be executed by the anonymization scheme due to its extends
    Anonymization_scheme class
    (See examples of use in section 5 of the jupyter notebook: test_anonymization.ipynb)
    (See also the file "test_differential_privacy.py" in the folder "tests")

    See Also
    --------
    :class:`Anonymization_scheme`

     References
    ----------
    .. [3] Jordi Soria-Comas, Josep Domingo-Ferrer, David Sánchez and Sergio Martínez, "Enhancing data utility in differential privacy via microaggregation-based k-anonymity", The VLDB Journal, Vol. 23, no. 5, pp. 771-794, Sep 2014. DOI: https://doi.org/10.1007/s00778-014-0351-4

    """

    def __init__(self, original_dataset, k, epsilon):
        """Constructor, called from inherited classes

        Parameters
        ----------
        original_dataset : :class:`Dataset`
            the data set to be anonymized.

        k : int
            The size of the clusters in the clustering process

        epsilon : float
            The desired level of differential privacy during the anonymization process

        See Also
        --------
        :class:`Dataset`
        """
        super().__init__(original_dataset)
        self.k = k
        self.epsilon = epsilon

    def calculate_anonymization(self, algorithm):
        """calculate_anonymization

        Function to perform the differential privacy anonymization.

        Parameters
        ----------
        algorithm : :class:`Algorithm`
            The clustering algorithm used during the anonymization.

        See Also
        --------
        :class:`Algorithm`
        """
        print("Anonymizing " + str(self) + " via " + str(algorithm))
        self.individual_ranking(algorithm)

    def individual_ranking(self, algorithm):
        t_ini = timer()
        self.anonymized_dataset = copy.deepcopy(self.original_dataset)
        reference_record_original = copy.copy(Record.reference_record)
        for i in range(self.original_dataset.num_attr):
            name = self.original_dataset.header[i]
            attribute = self.original_dataset.attributes[name]
            sensitivity = attribute.sensitivity_type
            if sensitivity != Sensitivity_type.QUASI_IDENTIFIER.value:
                continue
            print(f"Anonymizing attribute: {name} ({attribute.attribute_type})")
            # Creating a temporal list of records with only this attribute
            temp = []
            for record in self.original_dataset.records:
                rec = Record(record.id, [record.values[i]])
                temp.append(rec)
            # Individual ranking works on an attribute
            Record.reference_record = Record(0, [reference_record_original.values[i]])
            Record.calculate_distances_to_reference_record(temp)

            # if values can not be negative put 0 in min_value in xml settings
            min_value = self.original_dataset.attributes[name].min_value
            max_value = self.original_dataset.attributes[name].max_value
            if min_value == "" or max_value == "":
                values = [record.values[0] for record in temp]
                min_value_margin, max_value_margin = values[0].calculate_min_max(
                    values, constants.BORDER_MARGIN
                )
                if min_value == "":
                    min_value = min_value_margin
                if max_value == "":
                    max_value = max_value_margin
            applicable_epsilon = self.epsilon / self.original_dataset.num_attr_quasi

            clusters = algorithm.create_clusters(temp, self.k)
            for cluster in clusters:
                centroid = algorithm.calculate_centroid(
                    cluster,
                    epsilon=applicable_epsilon,
                    k=self.k,
                    min_value=min_value,
                    max_value=max_value,
                )
                for record in cluster:
                    self.anonymized_dataset.records[record.id].values[
                        i
                    ] = centroid.values[0]
        # this is to allow other anonymizations without re-load the dataset
        Dataset.calculate_standard_deviations(self.original_dataset.records)
        Record.set_reference_record(self.original_dataset)
        self.suppress_identifiers()
        Anonymization_scheme.runtime = timer() - t_ini
        print(
            f"Anonymization runtime: {utils.format_time(Anonymization_scheme.runtime)}"
        )

    def __str__(self):
        return (
            "Differential_privacy, k = "
            + str(self.k)
            + ", epsilon = "
            + str(self.epsilon)
        )
