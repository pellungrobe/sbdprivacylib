from abc import ABC, abstractmethod
from privlib.anonymization.src.entities.information_loss_result import (
    Information_loss_result,
)
from privlib.anonymization.src.utils.sensitivity_type import Sensitivity_type
import pandas as pd
from IPython.display import display
import copy
from privlib.anonymization.src.entities.disclosure_risk_result import (
    Disclosure_risk_result,
)
from privlib.anonymization.src.entities.dataset import Dataset
from privlib.anonymization.src.entities.dataset_SPF import Dataset_SPF
from privlib.anonymization.src.entities.record import Record
from tqdm.auto import tqdm
from privlib.anonymization.src.utils import utils
from privlib.anonymization.src.utils import constants


class Anonymization_scheme(ABC):
    """Anonymization_scheme

    Abstract class that represents the anonymization scheme.
    Defines a series of functions and attributes necessaries in all anonymization methods.
    Classes implementing an anonymization method must extend this class.
    (See examples of use in sections 1, 2 ,3 ,4 and 5 of the jupyter notebook: test_anonymization.ipynb)
    (See also the files "test_k_anonymity.py", "test_k_t_closeness.py" and "test_differential_privacy.py"
     in the folder "tests")

    References
    ----------
    .. [1] Josep Domingo-Ferrer and Vicenç Torra, "Ordinal, continuous and heterogeneous k-anonymity through microaggregation", Data Mining and Knowledge Discovery, Vol. 11, pp. 195-212, Sep 2005. DOI: https://doi.org/10.1007/s10618-005-0007-5
    .. [4] Josep Domingo-Ferrer and Vicenç Torra, "Disclosure risk assessment in statistical data protection", Journal of Computational and Applied Mathematics, Vol. 164, pp. 285-293, Mar 2004. DOI: https://doi.org/10.1016/S0377-0427(03)00643-5

    """

    def __init__(self, original_dataset):
        """Constructor, called from inherited classes

        Parameters
        ----------
        original_dataset : :class:`Dataset`
            the data set to be anonymized.

        See Also
        --------
        :class:`Dataset`
        """
        self.original_dataset = original_dataset
        self.anonymized_dataset = original_dataset
        self.runtime = 0

    @abstractmethod
    def calculate_anonymization(self, algorithm):
        """calculate_anonymization

        Function to perform the anonymization of the dataset given in the constructor
        Abstract method, all anonymization methods must implement it.

        Parameters
        ----------
        algorithm : :class:`Algorithm`
            the clustering algorithm used to group records during the anonymization.

        See Also
        --------
        :class:`Algorithm`
        """
        pass

    def suppress_identifiers(self):
        """suppress_identifiers

        Function that removes the identifiers attribute values from the data set.
        """
        for i in range(self.anonymized_dataset.num_attr):
            name = self.anonymized_dataset.header[i]
            attribute = self.anonymized_dataset.attributes[name]
            sensitivity = attribute.sensitivity_type
            if sensitivity == Sensitivity_type.IDENTIFIER.value:
                type_value = type(self.anonymized_dataset.records[0].values[i])
                for record in self.anonymized_dataset.records:
                    record.values[i].value = type_value.reference_value.value

    @staticmethod
    def calculate_information_loss(original_dataset, anonymized_dataset):
        """calculate_information_loss

        Function to perform the clustering of the records given as parameter
        Abstract method, all clustering algorithms must implement it.

        Parameters
        ----------
        original_dataset : :class:`Dataset`
            The original data set.

        anonymized_dataset : :class:`Dataset`
            The anonymized version of the original dataset
        Returns
        -------
        :class:`Information_loss_result`
            Information loss statistics.

        See Also
        --------
        :class:`Record`
        :class:`Information_loss_result`
        """
        print("Calculating information loss metrics")
        Dataset.calculate_standard_deviations(original_dataset.records)
        SSE = 0
        for i in range(len(original_dataset)):
            dis = original_dataset.records[i].distance(anonymized_dataset.records[i])
            SSE += dis
        SSE /= len(original_dataset)
        num_attr = len(original_dataset.records[0].values)
        attribute_name = []
        original_mean = []
        original_variance = []
        for i in range(num_attr):
            attribute_name.append(original_dataset.header[i])
            values = []
            for j in range(len(original_dataset)):
                values.append(original_dataset.records[j].values[i])
            mean = original_dataset.records[0].values[i].calculate_mean(values)
            original_mean.append(mean)
            variance = original_dataset.records[0].values[i].calculate_variance(values)
            original_variance.append(variance)
        anonymized_mean = []
        anonymized_variance = []
        for i in range(num_attr):
            values = []
            for j in range(len(anonymized_dataset)):
                values.append(anonymized_dataset.records[j].values[i])
            mean = anonymized_dataset.records[0].values[i].calculate_mean(values)
            anonymized_mean.append(mean)
            variance = (
                anonymized_dataset.records[0].values[i].calculate_variance(values)
            )
            anonymized_variance.append(variance)

        information_loss = Information_loss_result(
            SSE,
            attribute_name,
            original_mean,
            anonymized_mean,
            original_variance,
            anonymized_variance,
        )

        return information_loss

    def save_anonymized_dataset(self, path):
        """save_anonymized_dataset

        Function Called to save the anonymized dataset.

        Parameters
        ----------
        path : str
            desired path to save the anonymized dataset.
        """
        file = open(path, "w")
        file.write(
            Anonymization_scheme.list_to_string(
                self.anonymized_dataset.header, self.anonymized_dataset.separator
            )
        )
        file.write("\n")
        for record in self.anonymized_dataset.records:
            file.write(
                Anonymization_scheme.list_to_string(
                    record.values, self.anonymized_dataset.separator
                )
            )
            file.write("\n")
        file.close()
        display("Dataset saved: " + path)

    def anonymized_dataset_to_dataframe(self):
        """anonymized_dataset_to_dataframe

        Function Called to convert the anonymized dataset to a pandas dataframe.

        Returns
        -------
        DataFrame :DataFrame
            The pandas dataframe.
        """
        columns = self.anonymized_dataset.header
        data = []
        for record in self.anonymized_dataset.records:
            data.append(str(record).split(","))
        df = pd.DataFrame(data, columns=columns)

        return df

    def anonymized_dataset_to_SPF(self):
        """anonymized_dataset_to_SPF

        Function Called to convert the anonymized dataset to a :class:`SequentialPrivacyFrame`.

        Returns
        -------
        :class:`SequentialPrivacyFrame`
            The SequentialPrivacyFrame data set.
        """
        print(type(self.anonymized_dataset))
        if isinstance(self.anonymized_dataset, Dataset_SPF):
            for i in range(self.anonymized_dataset.num_attr):
                for j in range(len(self.anonymized_dataset)):
                    name = self.anonymized_dataset.header[i]
                    attribute = self.anonymized_dataset.attributes[name]
                    sensitivity = attribute.sensitivity_type
                    if (
                        sensitivity != Sensitivity_type.QUASI_IDENTIFIER.value
                        and sensitivity != Sensitivity_type.IDENTIFIER.value
                    ):
                        continue
                    value_ori = self.anonymized_dataset.spf.iloc[j, i]
                    if name == "elements":
                        self.anonymized_dataset.spf.drop(
                            "elements", inplace=True, axis=1
                        )
                        elements = []
                        for record in self.anonymized_dataset.records:
                            value_anom = record.values[i].value
                            value_anom = type(value_ori)(value_anom)
                            # tup = tuple(value_anom)
                            elements.append(value_anom)
                        self.anonymized_dataset.spf.insert(i, "elements", elements)
                        break
                    else:
                        value_anom = self.anonymized_dataset.records[j].values[i].value
                        value_anom = type(value_ori)(value_anom)
                        self.anonymized_dataset.spf.iloc[j, i] = value_anom
        else:
            raise TypeError(
                f"Sequential Privacy Frame format required for original dataset : "
                f"{type(self.original_dataset)}"
            )

        return self.anonymized_dataset.spf

    @staticmethod
    def calculate_record_linkage(original_dataset, anonymized_dataset):
        """calculate_record_linkage

        Function to Calculates the disclosure risk of the anonymized data set by comparing it with
        the original one.

        Parameters
        ----------
        original_dataset : :class:`Dataset`
            The original data set.

        anonymized_dataset : :class:`Dataset`
            The anonymized version of the original dataset
        Returns
        -------
        :class:`Disclosure_risk_result`
            The disclosure risk.

        See Also
        --------
        :class:`Disclosure_risk_result`
        """
        print("Calculating record linkage (disclosure risk)")
        Dataset.calculate_standard_deviations(original_dataset.records)
        control = {}
        ids = {}
        for record in original_dataset.records:
            count = control.get(record)
            if count is not None:
                count += 1
            else:
                count = 1
                ids[record] = []
            control[record] = count
            ids[record].append(record.id)

        total_prob = 0
        min_rec = None
        for record_anom in tqdm(anonymized_dataset.records):
            min_dist = float("inf")
            for record_ori in original_dataset.records:
                dist = record_ori.distance(record_anom)
                if dist < min_dist:
                    min_dist = dist
                    min_rec = copy.deepcopy(record_ori)
            ids_group = ids[min_rec]
            if record_anom.id in ids_group:
                count = control[min_rec]
                partial = 1 / count
                total_prob += partial

        return Disclosure_risk_result(total_prob, len(anonymized_dataset))

    @staticmethod
    def calculate_fast_record_linkage(
        original_dataset, anonymized_dataset, window_size=None
    ):
        """calculate_fast_record_linkage

        Function to Calculates the disclosure risk of the anonymized data set by comparing it with
        the original one. This is a fast version of record linkage calculation but less accurate

        Parameters
        ----------
        original_dataset : :class:`Dataset`
            The original data set.

        anonymized_dataset : :class:`Dataset`
            The anonymized version of the original dataset

        window_size : int
            optional, The desired size of the window, the greater the window the more accurate, but slower.
            If it is omitted, the 1% of the data set is taken
        Returns
        -------
        :class:`Disclosure_risk_result`
            The disclosure risk.

        See Also
        --------
        :class:`Disclosure_risk_result`
        """
        if window_size is None:
            window_size = (len(original_dataset) * constants.WINDOW_SIZE) / 100
            if window_size < 1.0:
                window_size = len(original_dataset)
        print(
            "Calculating fast record linkage (disclosure risk), window size = "
            + str(window_size)
        )
        Dataset.calculate_standard_deviations(original_dataset.records)
        Record.set_reference_record(original_dataset)
        Record.calculate_distances_to_reference_record(anonymized_dataset.records)
        control = {}
        ids = {}
        for record in original_dataset.records:
            count = control.get(record)
            if count is not None:
                count += 1
            else:
                count = 1
                ids[record] = []
            control[record] = count
            ids[record].append(record.id)

        original_dataset.records.sort(key=lambda x: x.distance_to_reference_record)
        distances = [
            record.distance_to_reference_record for record in original_dataset.records
        ]

        min_rec = None
        total_prob = 0
        for record_anom in tqdm(anonymized_dataset.records):
            closest_records = utils.take_closest_window(
                distances, record_anom.distance_to_reference_record, window_size
            )
            min_dist = float("inf")
            for pos in closest_records:
                record = original_dataset.records[pos]
                dist = record.distance_all_attributes(record_anom)
                if dist < min_dist:
                    min_dist = dist
                    min_rec = copy.deepcopy(record)
            ids_group = ids[min_rec]
            if record_anom.id in ids_group:
                count = control[min_rec]
                partial = 1 / count
                total_prob += partial

        # rearranging
        original_dataset.records.sort(key=lambda x: x.id)

        return Disclosure_risk_result(total_prob, len(anonymized_dataset))

    @staticmethod
    def list_to_string(list_to, separator):
        s = ""
        for value in list_to:
            s += str(value)
            s += separator
        s = s[: len(s) - 1]

        return s
