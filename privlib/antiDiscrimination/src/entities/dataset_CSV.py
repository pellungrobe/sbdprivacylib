from antiDiscrimination.src.entities.dataset import Dataset


class Dataset_CSV(Dataset):
    """Dataset_CSV

    Class that represents a dataset of records stored in csv format
    (See examples of use in sections 1 and 2 of the jupyter notebook: test_antiDiscrimination.ipynb)
    (See also the file "anti_discrimination_test.py" in the folder "tests")

    """
    def __init__(self, dataset_path, separator, sample=None):
        """Constructor, creates an instance of a dataset loaded from a csv formatted file

        Parameters
        ----------
        dataset_path :
            Path location of the dataset
        separator :
            The separator character of the csv file
        sample :
            Optional, Load only a random sample of size sample, if it is omitted, it is loaded the whole dataset

        See Also
        --------
        :class:`Dataset`
        """
        self.dataset_path = dataset_path
        self.name = dataset_path
        super().__init__(self.name, separator, sample)

    def load_header(self):
        """load_header

        Load the header of the dataset. The header consist of the name of the attributes

        See Also
        --------
        :class:`Dataset`
        """
        file = open(self.dataset_path, "r")
        header = file.readline()
        header = header.strip("\n")
        header = header.split(self.separator)
        file.close()
        self.set_header(header)

    def load_dataset(self):
        """load_dataset

        Load the dataset. Implements the inherited load_dataset method for the csv formatted file

        See Also
        --------
        :class:`Dataset`
        """
        file = open(self.dataset_path, "r")
        file.readline()  # Skip header
        for line in file:
            record_str = line.strip("\n")
            record_str = record_str.split(self.separator)
            super().add_record(record_str)
        file.close()

    def description(self):
        super().dataset_description()

    def __str__(self):
        return self.name
