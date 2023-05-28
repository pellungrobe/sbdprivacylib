from enum import Enum


class Attribute_type(Enum):
    """Attribute_type

    Enumeration which relates the available attribute types.
    The items enumerated consist of the name of the class implemented for each specific attribute type.
    Other classes implementing other attribute types can be added in the enumeration.
    These attribute type classes must implement the Value class interface
    (See examples of use in section 1 of the jupyter notebook: test_anonymization.ipynb)

    """
    # ["name","path", "module", "class"]
    # Current available attribute types:
    NUMERICAL_CONTINUOUS = ["numerical_continuous",
                            "anonymization.src.attribute_types.numerical_continuous",
                            "numerical_continuous",
                            "Numerical_continuous"]
    NUMERICAL_DISCRETE = ["numerical_discrete",
                          "anonymization.src.attribute_types.numerical_discrete",
                          "numerical_discrete",
                          "Numerical_discrete"]
    PLAIN_CATEGORICAL = ["plain_categorical",
                         "anonymization.src.attribute_types.plain_categorical",
                         "plain_categorical",
                         "Plain_categorical"]
    SEMANTIC_CATEGORICAL_WORDNET = ["semantic_categorical_wordnet",
                                    "anonymization.src.attribute_types.semantic_categorical_wordnet",
                                    "semantic_categorical_wordnet",
                                    "Semantic_categorical_wordnet"]
    DATE = ["date",
            "anonymization.src.attribute_types.date",
            "date",
            "Date"]
    COORDINATE = ["coordinate",
                  "anonymization.src.attribute_types.coordinate",
                  "coordinate",
                  "Coordinate"]
    DATETIME = ["datetime",
            "anonymization.src.attribute_types.datetime",
            "datetime",
            "Datetime"]
