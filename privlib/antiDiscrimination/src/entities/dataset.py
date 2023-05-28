from abc import ABC, abstractmethod
import pandas as pd
from IPython.display import display
from antiDiscrimination.src.entities.record import Record
import random


class Dataset(ABC):
    """Dataset

    Abstract class that represents a dataset of records
    Different dataset formats have to inherit this class
    (See examples of use in sections 1 and 2 of the jupyter notebook: test_antiDiscrimination.ipynb)
    (See also the file "anti_discrimination_test.py" in the folder "tests")

    """
    def __init__(self, name, separator, sample=None):
        """Constructor, creates an instance of a dataset

        Parameters
        ----------
        name :
            Name of the dataset, it consists of the file where is stored the dataset
        separator :
            The separator character of the csv file
        sample :
            Optional, Load only a random sample of size sample, if it is omitted, it is loaded the whole dataset

        """
        self.name = name
        self.separator = separator
        self.records = []
        self.header = []
        self.num_attr = 0
        self.load_header()
        self.load_dataset()
        if sample is not None:
            self.take_sample(sample)
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
        """
        for attribute_name in header:
            self.header.append(attribute_name)

    def add_record(self, values_in):
        """add_record

        Parameters
        ----------
        values_in :
            The record to be stored in the dataset. It consist of a list of values.

        Adds a record to the dataset. The load_dataset method implementation should call this method to store the data

        See Also
        --------
        :class:`Record`
        """
        record = Record(len(self.records), values_in)
        self.records.append(record)

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
            row = [name]
            table.append(row)
        df = pd.DataFrame(table, columns=["Name"])
        df.style.set_caption('Attributes')
        display(df)
        print("")

    def take_sample(self, sample):
        self.records = random.sample(self.records, sample)
        for id, record in enumerate(self.records):
            record.id = id

    def __len__(self):
        return len(self.records)

    def __str__(self):
        return self.name
