import pandas as pd
from IPython.display import display


class Information_loss_result:
    """Information_loss_result

    Class that stores the results of the information loss calculation.
    This class is used to store the information loss calculated with the
    calculate_information_loss method of class :class:`Anonymization_scheme`
    (See examples of use in sections 1, 2, 3, 4, and 5 of the jupyter notebook: test_anonymization.ipynb)
    (See also the files "test_k_anonymity.py", "test_k_t_closeness.py" and "test_differential_privacy.py"
     in the folder "tests")
    """
    def __init__(self, SSE, attribute_name, original_mean, anonymized_mean,
                 original_variance, anonymized_variance):
        """Constructor, creates an instance of a information loss result

        Parameters
        ----------
        SSE :
            The sum of square error result of the information loss calculation
        attribute_name :
            The list of names of the attributes
        original_mean :
            The list of means of each attribute of the original data set
        anonymized_mean :
            The list of means of each attribute of the anonymized data set
        original_variance :
            The list of variances of each attribute of the original data set
        anonymized_variance :
            The list of variances of each attribute of the anonymized data set

        See Also
        --------
        :class:`Anonymization_scheme`
        """
        self.SSE = SSE
        self.attribute_name = attribute_name
        self.original_variance = original_variance
        self.original_mean = original_mean
        self.anonymized_variance = anonymized_variance
        self.anonymized_mean = anonymized_mean

    def description(self):
        print("")
        print("Information loss metrics:")
        print(f"SSE: {self.SSE:.3f}")
        table = []
        for i in range(len(self.original_mean)):
            row = [self.attribute_name[i], self.original_mean[i], self.anonymized_mean[i],
                   self.original_variance[i], self.anonymized_variance[i]]
            table.append(row)
        df = pd.DataFrame(table, columns=["Name", "Original mean", "Anonymized mean",
                                          "Original variance", "Anonymized variance"])
        display(df)

    def __str__(self):
        s = "SSE: " + str(self.SSE) + "\n"
        for i in range(len(self.original_mean)):
            s += "Attribute: " + self.attribute_name[i] + " Original data set mean: " + str(
                self.original_mean[i]) + "\n"
            s += "Attribute: " + self.attribute_name[i] + " Anonymized data set mean: " + str(
                self.anonymized_mean[i]) + "\n"
            s += "Attribute: " + self.attribute_name[i] + " Original data set variance: " + str(
                self.original_variance[i]) + "\n"
            s += "Attribute: " + self.attribute_name[i] + " Anonymized data set variance: " + str(
                self.anonymized_variance[i]) + "\n"

        return s
