from enum import Enum


class Sensitivity_type(Enum):
    """Sensitivity_type

    Enumeration which relates the available sensitivity types
    """
    IDENTIFIER = "identifier"
    QUASI_IDENTIFIER = "quasi_identifier"
    CONFIDENTIAL = "confidential"
    NON_CONFIDENTIAL = "non_condifential"
