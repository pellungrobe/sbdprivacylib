from privlib.anonymization.src.entities.dataset_CSV import Dataset_CSV
from privlib.anonymization.src.entities.dataset_DataFrame import Dataset_DataFrame
from privlib.anonymization.src.algorithms.microaggregation import Microaggregation
from privlib.anonymization.src.algorithms.differential_privacy import Differential_privacy
from privlib.anonymization.src.utils import utils
from privlib.anonymization.src.algorithms.anonymization_scheme import Anonymization_scheme

"""
References:
[1] Josep Domingo-Ferrer and Vicenç Torra, "Ordinal, continuous and heterogeneous k-anonymity through microaggregation",
    Data Mining and Knowledge Discovery, Vol. 11, pp. 195-212, Sep 2005. DOI: https://doi.org/10.1007/s10618-005-0007-5
[3] Jordi Soria-Comas, Josep Domingo-Ferrer, David Sánchez and Sergio Martínez,
    "Enhancing data utility in differential privacy via microaggregation-based k-anonymity",
    The VLDB Journal, Vol. 23, no. 5, pp. 771-794, Sep 2014. DOI: https://doi.org/10.1007/s00778-014-0351-4
[4] Josep Domingo-Ferrer and Vicenç Torra, "Disclosure risk assessment in statistical data protection",
    Journal of Computational and Applied Mathematics, Vol. 164, pp. 285-293, Mar 2004.
    DOI: https://doi.org/10.1016/S0377-0427(03)00643-5

(See also detailed examples of use in section 5 of the jupyter notebook: test_anonymization.ipynb)
"""

""" Load a toy dataset containing all available datatypes. """

""" Load the data set and the xml describing them.
    Following, it is indicated the path to the csv file containing the data set and the path
    to the xml file describing the attributes in the data set.
    Inside the xml file, there is a detailed descritpion about how to fill this xml file
    in order to properly configure the different attribute types in the data set  """
# path_csv = "../../input_datasets/toy_all_types.csv"
# path_settings = "../../input_datasets/metadata_toy_all_types.xml"
# dataset = Dataset_CSV(path_csv, path_settings, ",")
path_csv = "../../input_datasets/toy_all_types.csv"
path_settings = "../../input_datasets/metadata_toy_all_types.xml"
data_frame = utils.read_dataframe_from_csv(path_csv)
# dataset = Dataset_CSV(path_csv, path_settings, ",")
dataset = Dataset_DataFrame(data_frame, path_settings)


""" Load the adult dataset containing 45222 records"""
# path_csv = "../../input_datasets/adult.csv"
# path_settings = "../../input_datasets/metadata_adult.xml"
# dataset = Dataset_CSV(path_csv, path_settings, ",")

"""Show a brief description of the loaded data set"""
dataset.description()

""" Select the desired anonymization method, clustering algorithm and its parameters
    and launch the anonymization process, in this tests, k-anonymity """
k = 3
epsilon = 1.0
anonymization_scheme = Differential_privacy(dataset, k, epsilon)
algorithm = Microaggregation()
anonymization_scheme.calculate_anonymization(algorithm)

""" Calculate information loss (utility) metrics and estimate the disclosure risk """
information_loss = Anonymization_scheme.calculate_information_loss(dataset, anonymization_scheme.anonymized_dataset)
information_loss.description()
disclosure_risk = Anonymization_scheme.calculate_fast_record_linkage(dataset, anonymization_scheme.anonymized_dataset)
disclosure_risk.description()

""" Save the anonymizated data set to disk in csv format"""
anonymization_scheme.save_anonymized_dataset("../../output_datasets/toy_all_types_anom.csv")

""" The anonymized data set can be converted to pandas dataframe"""
df_anonymized = anonymization_scheme.anonymized_dataset_to_dataframe()
print("Anonymized data set head: ")
print(df_anonymized.head())

""" Original data set and previously anonymized data set can be loaded 
    to calculate privacy metrics a posteriori """
# path_csv = "../../input_datasets/toy_all_types.csv"
# path_settings = "../../input_datasets/metadata_toy_all_types.xml"
# df = utils.read_dataframe_from_csv(path_csv)
# dataset_original = Dataset_DataFrame(df, path_settings)
#
# path_csv = "../../output_datasets/toy_all_types_anom.csv"
# path_settings = "../../input_datasets/metadata_toy_all_types.xml"
# df = utils.read_dataframe_from_csv(path_csv)
# dataset_anomymized = Dataset_DataFrame(df, path_settings)
#
# information_loss = Anonymization_scheme.calculate_information_loss(dataset_original, dataset_anomymized)
# information_loss.description()
# disclosure_risk = Anonymization_scheme.calculate_fast_record_linkage(dataset_original, dataset_anomymized)
# disclosure_risk.description()
