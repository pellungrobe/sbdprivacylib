from antiDiscrimination.src.algorithms.anti_discrimination import Anti_discrimination
from antiDiscrimination.src.entities.dataset_DataFrame import Dataset_DataFrame
from antiDiscrimination.src.utils import utils

"""
References:
[1] Sara Hajian and Josep Domingo-Ferrer, "A methodology for direct and indirect discrimination 
    prevention in data mining", IEEE Transactions on Knowledge and Data Engineering, Vol. 25, no. 7, pp. 1445-1459, 
    Jun 2013. DOI: https://doi.org/10.1109/TKDE.2012.72 

(See also examples of use in sections 1 and 2 of the jupyter notebook: test_antiDiscrimination.ipynb)
"""

""" Load the data set and the xml describing them """

""" Load the data set and the xml describing them.
    Following, it is indicated the path to the csv file containing the data set and the path
    to the xml file describing the attributes in the data set.
    Inside the xml file, there is a detailed descritpion about how to fill this xml file
    in order to properly configure the different attribute types in the data set  """

path_csv = "../../input_datasets/adult_anti_discrimination.csv"

""" Load the adult dataset containing 45222 records"""
data_frame = utils.read_dataframe_from_csv(path_csv)

dataset = Dataset_DataFrame(data_frame)

"""Show a brief description of the loaded data set"""
dataset.description()

""" Set the anti discrimination parameters:
    min support and min confidence: to consider a rule a frequent rule
    alfa: discriminatory threshold to consider a frequent rule a direct or an indirect discrimination rule
    DI: Predetermined discriminatory items (attribute,discriminatory value)"""
min_support = 0.02
min_confidence = 0.1
# alfa = 1.20
alfa = 1.0
DI = [("age","young"), ("sex","female")]
anonymization_scheme = Anti_discrimination(dataset, min_support, min_confidence, alfa, DI)

""" Launch the direct and indirect anti discrimination process """
anonymization_scheme.calculate_anonymization()
# sys.exit()

""" Calculate discrimination metrics """
anti_discrimination_metrics = anonymization_scheme.calculate_metrics()
anti_discrimination_metrics.description()

""" Save the anonymizated data set to disk in csv format"""
anonymization_scheme.save_anonymized_dataset("../../output_datasets/adult_anti_discrimination_anom.csv")

""" The anonymized data set can be converted to pandas dataframe"""
df_anonymized = anonymization_scheme.anonymized_dataset_to_dataframe()
print("Anonymized data set head: ")
print(df_anonymized.head())