from abc import ABC, abstractmethod
import pandas as pd
from IPython.display import display
from antiDiscrimination.src.entities.dataset import Dataset


class Anonymization_scheme(ABC):
    """Anonymization_scheme

    Abstract class that represents the anonymization scheme.
    Defines a series of functions and attributes necessaries in all anonymization methods.
    Classes implementing an anonymization method must extend this class.
    (See examples of use in sections 1 and 2 of the jupyter notebook: "test_antiDiscrimination.ipynb")
    (See also the file "anti_discrimmination_test.py" in the folder "tests")

    References
    ----------
    .. [1] Sara Hajian and Josep Domingo-Ferrer, "A methodology for direct and indirect discrimination prevention in
           data mining", IEEE Transactions on Knowledge and Data Engineering, Vol. 25, no. 7, pp. 1445-1459, Jun 2013.
           DOI: https://doi.org/10.1109/TKDE.2012.72

    """
    def __init__(self, original_dataset):
        """Constructor, called from inherited classes

        Parameters
        ----------
        original_dataset : :class:`Dataset`
            the data set to be anonymized (anti-discrimination).

        See Also
        --------
        :class:`Dataset`
        """
        self.original_dataset = original_dataset
        self.anonymized_dataset = original_dataset
        self.runtime = 0

    @abstractmethod
    def calculate_anonymization(self):
        """calculate_anonymization

        Function to perform the anonymization (anti-discrimination) of the dataset given in the constructor
        Abstract method, all anonymization methods must implement it.
         """
        pass

    @abstractmethod
    def calculate_metrics(self):
        """calculate_metrics

        Function to calculate the metrics of the datasets given in the constructor
        Abstract method, all anonymization methods must implement it.
        """
        pass

    def save_anonymized_dataset(self, path):
        """save_anonymized_dataset

        Function Called to save the anonymized dataset.

        Parameters
        ----------
        path : str
            desired path to save the anonymized dataset.
        """
        file = open(path, "w")
        file.write(Anonymization_scheme.list_to_string(self.anonymized_dataset.header,
                                                       self.anonymized_dataset.separator))
        file.write("\n")
        for record in self.anonymized_dataset.records:
            file.write(Anonymization_scheme.list_to_string(record.values,
                                                           self.anonymized_dataset.separator))
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

    @staticmethod
    def list_to_string(list_to, separator):
        s = ""
        for value in list_to:
            s += str(value)
            s += separator
        s = s[:len(s) - 1]

        return s
