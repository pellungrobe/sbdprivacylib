
class Record:
    """Record

    Class that represents a record. A record consist of a list of values.
    A :class:`Dataset` is formed by a list of records
    """
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

    def __str__(self):
        s = ""
        for value in self.values:
            s += str(value)
            s += ","
        s = s[:len(s) - 1]

        return s
