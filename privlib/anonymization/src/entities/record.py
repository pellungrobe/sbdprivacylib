import math
from anonymization.src.utils.sensitivity_type import Sensitivity_type


class Record:
    """Record

    Class that represents a record. A record consist of a list of values.
    A :class:`Dataset` is formed by a list of records
    """
    header = []
    attributes = {}
    standard_deviations = []
    reference_record = None

    def __init__(self, id_rec, values):
        """Constructor, creates an instance of a record

        Parameters
        ----------
        id_rec : int
            The identifier of this record
        values : list
            The list of values that form the record (a value for attribute)

        """
        self.id = id_rec
        self.values = values
        self.distance_to_reference_record = 0

    def distance(self, record):
        """Calculates the distance between this record and the record given as parameter.
        The distance is calculated as the Euclidean distance normalized by standard deviation.
        The distance of each attribute value is calculated with each specific attribute value implementation
        This method takes only the quasi-identifier attributes to calculate the distance between records.

        Parameters
        ----------
        record : :class:`Record`
            The record to calculate the distance

        Returns
        -------
        float
            The distance between this record and the given one.

        """
        # Euclidean distance normalized by standard deviation
        partial = 0
        num_quasi = 0
        for i in range(len(self.values)):
            name = Record.header[i]
            sensitivity_type = Record.attributes[name].sensitivity_type
            # Taking into account only quasi_identifiers
            if sensitivity_type == Sensitivity_type.QUASI_IDENTIFIER.value:
                distance = self.values[i].distance(record.values[i])
                distance /= Record.standard_deviations[i]
                partial += distance * distance
                num_quasi += 1
        partial /= num_quasi
        distance = math.sqrt(partial)

        return distance

    def distance_all_attributes(self, record):
        """Calculates the distance between this record and the record given as parameter.
        The distance is calculated as the Euclidean distance normalized by standard deviation.
        The distance of each attribute value is calculated with each specific attribute value implementation
        This method takes all attributes to calculate the distance between records.

        Parameters
        ----------
        record : :class:`Record`
            The record to calculate the distance

        Returns
        -------
        float
            The distance between this record and the given one.

        """
        # Euclidean distance normalized by standard deviation
        partial = 0
        for i in range(len(self.values)):
            distance = self.values[i].distance(record.values[i])
            distance /= Record.standard_deviations[i]
            partial += distance * distance
        partial /= len(self.values)
        distance = math.sqrt(partial)

        return distance

    @staticmethod
    def set_reference_record(dataset):
        """Creates and stored a record that is formed by the reference value of each attribute.

        Parameters
        ----------
        dataset : :class:`Dataset`
            The dataset to create the reference record

        """
        reference_values = []
        for i in range(dataset.num_attr):
            name = dataset.header[i]
            sensitivity_type = dataset.attributes[name].sensitivity_type
            if sensitivity_type == Sensitivity_type.QUASI_IDENTIFIER.value or \
               sensitivity_type == Sensitivity_type.IDENTIFIER.value:
                values = []
                for j in range(len(dataset)):
                    values.append(dataset.records[j].values[i])
                reference_value = values[0].calculate_reference_value(values)
                reference_values.append(reference_value)
            else:
                reference_values.append(None)
        Record.reference_record = Record(0, reference_values)
        Record.calculate_distances_to_reference_record(dataset.records)

    @staticmethod
    def calculate_distances_to_reference_record(records):
        """Calculates and stores for each record in the list given as parameter the distance to the reference record.

        Parameters
        ----------
        records : list
            The list of records to calculate and store the distance to the reference record

        """
        for record in records:
            record.distance_to_reference_record = record.distance(Record.reference_record)

    def __copy__(self):
        return Record(0, self.values)

    def __eq__(self, other):
        distance1 = self.distance_to_reference_record
        distance2 = other.distance_to_reference_record

        return distance1 == distance2

    def __hash__(self):
        return hash(str(self))

    def __lt__(self, other):
        distance1 = self.distance_to_reference_record
        distance2 = other.distance_to_reference_record

        return distance1 < distance2

    def __str__(self):
        s = ""
        for value in self.values:
            s += str(value)
            s += ","
        s = s[:len(s) - 1]

        return s
