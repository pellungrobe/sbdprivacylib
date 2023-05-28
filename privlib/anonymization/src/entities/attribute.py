
class Attribute:
    """Attribute

    Class that represents an attribute in the dataset metadata.
    This class stores the information about an attribute in the dataset.

    """
    def __init__(self, name, attribute_type, sensitivity_type,
                 min_value, max_value):
        """Constructor, creates an instance of an attribute

        Parameters
        ----------
        name :
            the name of the attribute
        attribute_type :
            the type of the attribute, it should be an item of Attribute_type enumeration
        sensitivity_type :
            the sensitivity of the attribute, it should be an item of Sensitivity_Type enumeration
        min_value :
            the minimum value of the attribute in its domain
        max_value :
            the maximum value of the attribute in its domain

        See Also
        --------
        :class:`Attribute_type`
        :class:`Sensitivity_Type`
        """
        self.name = name
        self.attribute_type = attribute_type
        self.sensitivity_type = sensitivity_type
        self.min_value = min_value
        self.max_value = max_value

    def __str__(self):
        s = self.name
        s += " (" + self.attribute_type + " / "
        s += self.sensitivity_type + ")"

        return s
