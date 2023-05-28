from abc import ABC, abstractmethod
from xml.dom import minidom
import pandas as pd
from IPython.display import display
from privlib.anonymization.src.entities.attribute import Attribute
from privlib.anonymization.src.utils.utils import get_class
from privlib.anonymization.src.attribute_types.attribute_type import Attribute_type
from privlib.anonymization.src.entities.record import Record
from privlib.anonymization.src.utils.sensitivity_type import Sensitivity_type
from privlib.anonymization.src.utils import constants
import random


class Dataset(ABC):
    """Dataset

    Abstract class that represents a dataset of records and described by the metadata stored in settings_path or
    attrs_settings
    Different dataset formats have to inherit this class
    (See examples of use in sections 1, 2, 3, 4, and 5 of the jupyter notebook: test_anonymization.ipynb)
    (See also the files "test_k_anonymity.py", "test_k_t_closeness.py" and "test_differential_privacy.py"
     in the folder "tests")

    """

    def __init__(self, name, settings_path, attrs_settings, separator, sample=None):
        """Constructor, creates an instance of a dataset

        Parameters
        ----------
        name :
            Name of the dataset, it consists of the file where is stored the dataset
        settings_path :
            The path of the xml file where it is stored the dataset metadata description
        attrs_settings :
            Alternatively, the metadata describing the dataset can be hardcoded
        separator :
            The separator character of the csv file
        sample :
            Optional, Load only a random sample of size sample, if it is omitted, it is loaded the whole dataset

        See Also
        --------
        :class:`Value`
        """
        self.name = name
        self.settings_path = settings_path
        self.attrs_settings = attrs_settings
        self.separator = separator
        self.attributes = {}
        self.records = []
        self.header = []
        self.available_attribute_types = {}
        self.num_attr = 0
        self.num_attr_quasi = 0
        self.load_dataset_settings()
        self.load_available_attribute_types()
        self.load_header()
        self.load_dataset()
        if sample is not None:
            self.take_sample(sample)
        Dataset.calculate_standard_deviations(self.records)
        self.set_reference_record()
        print("Dataset loaded: " + self.name)
        print("Records loaded: " + str(len(self)))

    @abstractmethod
    def load_header(self):
        """load_header

        Load the header of the dataset. The header consist of the name of the attributes
        """
        pass

    @abstractmethod
    def load_dataset(self):
        """load_dataset

        Load the dataset. The specific implementation should call the add_record method.
        """
        pass

    @abstractmethod
    def description(self):
        """description

        Shows a description of the dataset.
        """
        pass

    def set_header(self, header):
        """set_header

        Shows a description of the dataset.

        Parameters
        ----------
        header :
            The header including the name of the attributes in the dataset.
            The attribute names have to be included in the dataset metadata description
        """
        for attribute_name in header:
            if attribute_name in self.attributes:
                self.header.append(attribute_name)
            else:
                raise ValueError(
                    "("
                    + attribute_name
                    + ") "
                    + "Attribute name in header not found in settings file"
                )
        Record.header = self.header

    def load_available_attribute_types(self):
        """load_available_attribute_types

        Loads all available attribute types.

        See Also
        --------
        :class:`Attribute_type`
        """
        for attribute_type in Attribute_type:
            name = attribute_type.value[0]
            self.available_attribute_types[name] = attribute_type.value

    def add_record(self, values_in):
        """add_record

        Parameters
        ----------
        values_in :
            The record to be stored in the dataset. It consist of a list of values.
            The name of each value matches with the header and the attribute type is defined in the metadata

        Adds a record to the dataset. The load_dataset method implementation should call this method to store the data

        See Also
        --------
        :class:`Record`
        """
        values = []
        for i in range(len(values_in)):
            print(self.available_attribute_types)
            attribute = self.attributes[self.header[i]]
            path = self.available_attribute_types[attribute.attribute_type][1]
            module = self.available_attribute_types[attribute.attribute_type][2]
            class_type = self.available_attribute_types[attribute.attribute_type][3]
            value = values_in[i]
            instance = get_class(path, module, class_type, value)
            values.append(instance)
        record = Record(len(self.records), values)
        self.records.append(record)

    def load_dataset_settings(self):
        """load_dataset_settings

        Loads the dataset metadata describing each attribute type
        """
        if self.settings_path is not None:
            doc = minidom.parse(self.settings_path)
            attributes_setting = doc.getElementsByTagName(constants.ATTRIBUTE)
            for attribute in attributes_setting:
                name = attribute.getAttribute("name")
                sensitivity_type = attribute.getAttribute(constants.SENSITIVITY_TYPE)
                attribute_type = attribute.getAttribute(constants.ATTRIBUTE_TYPE)
                min_value = attribute.getAttribute(constants.MIN_VALUE)
                max_value = attribute.getAttribute(constants.MAX_VALUE)
                attribute = Attribute(
                    name, attribute_type, sensitivity_type, min_value, max_value
                )
                self.attributes[name] = attribute
                self.num_attr += 1
                if sensitivity_type == Sensitivity_type.QUASI_IDENTIFIER.value:
                    self.num_attr_quasi += 1
        elif self.attrs_settings is not None:
            for name in self.attrs_settings.keys():
                sensitivity_type = self.attrs_settings[name][constants.SENSITIVITY_TYPE]
                attribute_type = self.attrs_settings[name][constants.ATTRIBUTE_TYPE][0]
                min_value = ""
                max_value = ""
                if constants.MIN_VALUE in self.attrs_settings[name].keys():
                    min_value = self.attrs_settings[name][constants.MIN_VALUE]
                if constants.MAX_VALUE in self.attrs_settings[name].keys():
                    max_value = self.attrs_settings[name][constants.MAX_VALUE]
                attribute = Attribute(
                    name, attribute_type, sensitivity_type, min_value, max_value
                )
                self.attributes[name] = attribute
                self.num_attr += 1
                if sensitivity_type == Sensitivity_type.QUASI_IDENTIFIER.value:
                    self.num_attr_quasi += 1
        Record.attributes = self.attributes

    def dataset_description(self):
        """dataset_description

        Shows a description of the dataset using pandas Dataframe
        """
        print("Dataset description:")
        print("Data set: " + self.name)
        print("Records: " + str(len(self)))
        print("Attributes:")
        table = []
        for name in self.header:
            atribute = self.attributes[name]
            row = [atribute.name, atribute.attribute_type, atribute.sensitivity_type]
            table.append(row)
        df = pd.DataFrame(table, columns=["Name", "Attribute_type", "Sensitivity_type"])
        df.style.set_caption("Attributes")
        display(df)
        print("")

    @staticmethod
    def calculate_standard_deviations(records):
        """calculate_standard_deviations

        Calculates the standard deviations of the list of records given as parameter

        Parameters
        ----------
        records :
            The list of records to calculate the standard deviation.
            It is applied the specific value standard deviation calculation in function of the specific implementation.
            It is used to normalize values
        """
        Record.standard_deviations = []
        for i in range(len(records[0].values)):
            attr_data = []
            for j in range(len(records)):
                attr_data.append(records[j].values[i])
            standard_deviation = (
                records[0].values[i].calculate_standard_deviation(attr_data)
            )
            Record.standard_deviations.append(standard_deviation)

    def take_sample(self, sample):
        self.records = random.sample(self.records, sample)
        for id, record in enumerate(self.records):
            record.id = id

    def set_reference_record(self):
        Record.set_reference_record(self)

    def __len__(self):
        return len(self.records)

    def __str__(self):
        return self.name
