class Disclosure_risk_result:
    """Disclosure_risk_result

    Class that stores the results of the disclosure risk calculation
    (See examples of use in sections 1, 2, 3, 4, and 5 of the jupyter notebook: test_anonymization.ipynb)
    (See also the files "test_k_anonymity.py", "test_k_t_closeness.py" and "test_differential_privacy.py"
     in the folder "tests")
    """
    def __init__(self, disclosure_risk, dataset_size):
        """Constructor, creates an instance of a disclosure risk result

        Parameters
        ----------
        disclosure_risk :
            The result of the disclosure risk calculation
        dataset_size :
            The size of the data set of which the disclosure risk has been calculated

        See Also
        --------
        :class:`Anonymization_scheme`
        """
        self.disclosure_risk = disclosure_risk
        self.percen = (self.disclosure_risk / dataset_size) * 100

    def description(self):
        """Shows the results description of the disclosure risk calculation

        """
        print(f"Disclosure risk: {self.disclosure_risk:.3f} ({self.percen:.2f}%)")
