from antiDiscrimination.src.entities.dataset import Dataset
from IPython.display import display


class Dataset_DataFrame(Dataset):
    """Dataset_DataFrame

    Class that represents a dataset of records stored in pandas Dataframe format
    (See examples of use in sections 1 and 2 of the jupyter notebook: test_antiDiscrimination.ipynb)
    (See also the files "anti_discrimination_test.py" in the folder "tests")

    """
    def __init__(self, dataframe, sample=None):
        """Constructor, creates an instance of a dataset loaded from a pandas Dataframe

        Parameters
        ----------
        dataframe :
            The dataframe that stores the data set
        sample :
            Optional, Load only a random sample of size sample, if it is omitted, it is loaded the whole dataset

        See Also
        --------
        :class:`Dataset`
        """
        self.df = dataframe
        print("Loading dataset")
        super().__init__(self.df.name, ",", sample)

    def load_header(self):
        """load_header

        Load the header of the dataset. The header consist of the name of the attributes

        See Also
        --------
        :class:`Dataset`
        """
        header = self.df.columns.values.tolist()
        self.set_header(header)
        self.num_attr = len(header)

    def load_dataset(self):
        """load_dataset

        Load the dataset. Implements the inherited load_dataset method for the pandas Dataframe format

        See Also
        --------
        :class:`Dataset`
        """
        for i in range(len(self.df)):
            row = self.df.iloc[i].values
            super().add_record(row)

    def description(self):
        print("Dataset: " + self.name)
        print("Dataset head:")
        display(self.df.head())
        print("")
        super().dataset_description()

    def __str__(self):
        return self.name
