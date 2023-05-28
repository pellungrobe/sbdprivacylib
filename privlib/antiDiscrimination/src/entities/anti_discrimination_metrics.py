class Anti_discrimination_metrics:
    """Anti_discrimination_result

    Class that stores the results of the anti-discrimination metrics calculation
    (See examples of use in sections 1 and 2 of the jupyter notebook: test_antiDiscrimination.ipynb)
    (See also the file "anti_discrimination_test.py" in the folder "tests")
    """
    def __init__(self, DDPD, DDPP, IDPD, IDPP):
        """Constructor, creates an instance of an anti-discrimination metrics result

        Parameters
        ----------
        DDPD :
            The result of the direct discrimination prevention degree (DDPD) calculation
        DDPP :
            The result of the direct discrimination protection preservation (DDPP) calculation
        IDPD :
            The result of the indirect discrimination prevention degree (IDPD) calculation
        IDPP :
            The result of the indirect discrimination protection preservation (IDPP) calculation

        See Also
        --------
        :class:`Anonymization_scheme`
        """
        self.DDPD = DDPD
        self.DDPP = DDPP
        self.IDPD = IDPD
        self.IDPP = IDPP

    def description(self):
        """Shows the results description of the anti discrimination metrics calculation
        """
        print(f"DDPD: {self.DDPD:.2f}")
        print(f"DDPP: {self.DDPP:.2f}")
        print(f"IDPD: {self.IDPD:.2f}")
        print(f"IDPP: {self.IDPP:.2f}")

    def __str__(self):
        s = "DDPD: " + round(self.DDPD, 2) + "\n"
        s += "DDPP: " + round(self.DDPP, 2) + "\n"
        s += "IDPD: " + round(self.IDPD, 2) + "\n"
        s += "IDPP: " + round(self.IDPP, 2) + "\n"

        return s



