from anonymization.src.entities.dataset_SPF import Dataset_SPF
from anonymization.src.algorithms.mdav import Mdav
from anonymization.src.algorithms.microaggregation import Microaggregation
from anonymization.src.algorithms.k_anonymity import K_anonymity
from anonymization.src.algorithms.differential_privacy import Differential_privacy
from anonymization.src.algorithms.t_closeness import T_closeness
from anonymization.src.algorithms.anonymization_scheme import Anonymization_scheme
from risk_assessment.sequentialprivacyframe import SequentialPrivacyFrame
from anonymization.src.utils import constants
from anonymization.src.utils.sensitivity_type import Sensitivity_type
from anonymization.src.attribute_types.attribute_type import Attribute_type

"""
(See also examples of use in section 7 of the jupyter notebook: test_anonymization.ipynb)
"""

""" Load the spf toy dataset"""

""" Load the data set and the xml describing them.
    Following, it is indicated the path to the csv file containing the data set and the path
    to the xml file describing the attributes in the data set.
    Inside the xml file, there is a detailed descritpion about how to fill this xml file
    in order to properly configure the different attribute types in the data set  """
path_csv = "../../input_datasets/ToyDataset.txt"

"""The metadata can be read from a xml file describing the attributes in the data set"""
"""In this case, the metadata will be automatically embedded into the dataframe"""
# path_settings = "../../input_datasets/metadata_ToyDataset.xml"

"""Sequential Privacy Frame data is created from the csv dataset"""
spf = SequentialPrivacyFrame.from_file(path_csv, elements=["lat", "lng"], sequence_id="seq")

"""The metadata describing the attributes can be hardcoded (instead of read from xml)"""
settings = {"elements": {constants.SENSITIVITY_TYPE: Sensitivity_type.QUASI_IDENTIFIER.value,
                         constants.ATTRIBUTE_TYPE: Attribute_type.COORDINATE.value},
            "datetime": {constants.SENSITIVITY_TYPE: Sensitivity_type.QUASI_IDENTIFIER.value,
                         constants.ATTRIBUTE_TYPE: Attribute_type.DATETIME.value},
            "uid":      {constants.SENSITIVITY_TYPE: Sensitivity_type.IDENTIFIER.value,
                         constants.ATTRIBUTE_TYPE: Attribute_type.NUMERICAL_DISCRETE.value},
            "sequence": {constants.SENSITIVITY_TYPE: Sensitivity_type.IDENTIFIER.value,
                         constants.ATTRIBUTE_TYPE: Attribute_type.NUMERICAL_DISCRETE.value},
            "order":    {constants.SENSITIVITY_TYPE: Sensitivity_type.IDENTIFIER.value,
                         constants.ATTRIBUTE_TYPE: Attribute_type.NUMERICAL_DISCRETE.value}}

"""The attribute Dataframe.attrs has persistence and it is a good way to embed metadata into the dataframe"""
spf.attrs["attrs_settings"] = settings
# dataset = Dataset_SPF(spf, path_settings=path_settings)
dataset = Dataset_SPF(spf, attrs_settings=settings)
print(spf.attrs["attrs_settings"])

"""Show a brief description of the loaded data set"""
dataset.description()

""" Select the desired anonymization method, clustering algorithm and its parameters
    and launch the anonymization process"""
k = 5
anonymization_scheme = K_anonymity(dataset, k)
# epsilon = 1
# anonymization_scheme = Differential_privacy(dataset, k, epsilon)
# t = 0.25
# anonymization_scheme = T_closeness(dataset, k, t)
algorithm = Mdav()
# algorithm = Microaggregation()
anonymization_scheme.calculate_anonymization(algorithm)

""" Calculate information loss (utility) metrics and estimate the disclosure risk """
information_loss = Anonymization_scheme.calculate_information_loss(dataset, anonymization_scheme.anonymized_dataset)
information_loss.description()
# disclosure_risk = Anonymization_scheme.calculate_record_linkage(dataset, anonymization_scheme.anonymized_dataset)
# disclosure_risk.description()
disclosure_risk = Anonymization_scheme.calculate_fast_record_linkage(dataset,
                                                                     anonymization_scheme.anonymized_dataset)
disclosure_risk.description()

"""The anonymized dataset can be reverted to sequential privacy frame format"""
spf_anom = anonymization_scheme.anonymized_dataset_to_SPF()

""" Save the anonymizated data set to disk in csv format"""
# anonymization_scheme.save_anonymized_dataset("../../output_datasets/toy_all_types_anom.csv")

""" The anonymized data set can be converted to pandas dataframe"""
# df_anonymized = anonymization_scheme.anonymized_dataset_to_dataframe()
# print("Anonymized data set head: ")
# print(df_anonymized.head())
