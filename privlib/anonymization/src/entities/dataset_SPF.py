from privlib.anonymization.src.entities.dataset import Dataset
from privlib.anonymization.src.utils import constants
from IPython.display import display


class Dataset_SPF(Dataset):
    """Dataset_SPF

    Class that represents a dataset of records stored in SPF (sequential privacy frame) object format
    (See examples of use in section 7 of the jupyter notebook: test_anonymization.ipynb)
    (See also the file "test_spf.py" in the folder "tests")

    """

    def __init__(self, spf, path_settings=None, attrs_settings=None, sample=None):
        """Constructor, creates an instance of a dataset loaded from a SPF (sequential privacy frame) object

        Parameters
        ----------
        spf :
            The SPF that stores the data set
        path_settings :
            The path of the xml file where it is stored the dataset metadata description
        attrs_settings :
            Alternatively, The dataset metadata description can be hardcoded and loaded with this parameter
        sample :
            Optional, Load only a random sample of size sample, if it is omitted, it is loaded the whole dataset

        See Also
        --------
        :class:`Dataset`
        :class:`SequentialPrivacyFrame`
        """
        self.spf = spf
        self.settings_path = path_settings
        self.attrs_settings = attrs_settings
        print("Loading dataset")
        super().__init__("spf", path_settings, attrs_settings, ",", sample)
        self.add_attrs_settings_to_spf()

    def load_header(self):
        """load_header

        Implements the inherited load_header method for the SPF format
        Load the header of the dataset. The header consist of the name of the attributes

        See Also
        --------
        :class:`Dataset`
        :class:`SequentialPrivacyFrame`
        """
        header = self.spf.columns.values.tolist()
        self.set_header(header)

    def load_dataset(self):
        """load_dataset

        Load the dataset. Implements the inherited load_dataset method for the SPF format

        See Also
        --------
        :class:`Dataset`
        :class:`SequentialPrivacyFrame`
        """
        for i in range(len(self.spf)):
            row = self.spf.iloc[i].values
            super().add_record(row)

    def add_attrs_settings_to_spf(self):
        """add_attrs_settings_to_spf

        Adds the attribute description settings into the SPF

        See Also
        --------
        :class:`SequentialPrivacyFrame`
        """
        settings = {}
        if self.settings_path is not None:
            for name in self.header:
                attribute = self.attributes[name]
                settings[attribute.name] = {
                    constants.SENSITIVITY_TYPE: attribute.sensitivity_type,
                    constants.ATTRIBUTE_TYPE: self.available_attribute_types[
                        attribute.attribute_type
                    ],
                }
            self.spf.attrs["attrs_settings"] = settings
        elif self.attrs_settings is not None:
            self.spf.attrs["attrs_settings"] = self.attrs_settings

    def description(self):
        print("Dataset: " + self.name)
        print("Dataset head:")
        display(self.spf.head())
        print("")
        super().dataset_description()

    def __str__(self):
        return self.name
