import numpy as np
import pandas as pd
import pandas.core

from . import constants

__all__ = ["SequentialPrivacyFrame"]


class SequentialSeries(pd.Series):
    @property
    def _constructor(self):
        return SequentialSeries

    @property
    def _constructor_expanddim(self):
        return SequentialPrivacyFrame


class SequentialPrivacyFrame(pd.DataFrame):
    """SequentialPrivacyFrame.

    A SequentialPrivacyFrame object is a pandas.DataFrame that represents sequences.
    A sequence has at least the following attributes: user_id, datetime, order_id, sequence_id, elements.

    Parameters
    ----------
    data : list or dict or pandas DataFrame
        the data that must be embedded into a SequentialPrivacyFrame.

    user_id : int or str, optional
        the position or the name of the column in `data`containing the user identifier. The default is `constants.UID`.

    datetime : int or str, optional
        the position or the name of the column in `data` containing the datetime. The default is `constants.DATETIME`.

    order_id : int or str, optional
        the position or the name of the column in `data` containing the order identifier for the sequences.
        The default is `constants.ORDER_ID`.

    sequence_id : int or str, optional
        the position or the name of the column in `data` containing the sequence identifier.
        The default is `constants.SEQUENCE_ID`.

    elements : int or str, or list of int or list of str
        the positions or the names of the columns in  `data` containingn the elements of the sequences. Elements can be
        represented by any number of attributes that will be grouped together to represent the single element of the sequence.
        The default is `constants.ELEMENTS`

    timestamp : boolean, optional
        if True, the datetime is a timestamp. The default is `False`.

    check_order_date
        if True, the order of the various elements in the sequences of each user will be checked against the timestamp
        to ensure consistency. If some ordering attributes were not present in the original data, they will be computed
        based on what is available in the data. The default is `True`.
    """

    def __init__(
        self,
        data,
        user_id=constants.USER_ID,
        datetime=constants.DATETIME,
        order_id=constants.ORDER_ID,
        sequence_id=constants.SEQUENCE_ID,
        elements=constants.ELEMENTS,
        timestamp=False,
        check_order_date=True,
    ):
        d_columns = {
            user_id: constants.USER_ID,
            datetime: constants.DATETIME,
            sequence_id: constants.SEQUENCE_ID,
            order_id: constants.ORDER_ID,
        }

        columns = None
        if isinstance(data, pd.DataFrame):
            spf = data.rename(columns=d_columns)
            columns = spf.columns

        elif isinstance(data, dict):
            spf = pd.DataFrame(data).rename(columns=d_columns)
            columns = spf.columns

        elif isinstance(data, list) or isinstance(data, np.ndarray):
            columns = []
            num_columns = len(data[0])
            for i in range(num_columns):
                try:
                    columns += [d_columns[i]]
                except KeyError:
                    columns += [i]
            spf = pd.DataFrame(data, columns=columns)

        elif isinstance(data, pd.core.internals.BlockManager):
            spf = data

        else:
            raise TypeError(
                f"PrivacyDataFrame constructor called with incompatible data and dtype: {type(data)}"
            )
        if not isinstance(data, pd.core.internals.BlockManager):
            if isinstance(elements, str) or isinstance(elements, int):
                spf.rename(columns={elements: constants.ELEMENTS})
            elif isinstance(elements, list) and (
                all(isinstance(x, str) for x in elements)
                or all(isinstance(x, int) for x in elements)
            ):
                l = []
                for i in elements:
                    l.append(spf[i])
                spf[constants.ELEMENTS] = list(zip(*l))
                spf.drop(elements, axis="columns", inplace=True)
            else:
                raise TypeError(
                    f"PrivacyDataFrame constructor found elements of type: {type(elements)}"
                )
            columns = spf.columns

        super(SequentialPrivacyFrame, self).__init__(spf, columns=columns)

        if not isinstance(data, pd.core.internals.BlockManager):
            if check_order_date:
                self._check_order_date()
            if timestamp:
                self[constants.DATETIME] = pd.to_datetime(
                    self[constants.DATETIME], unit="s"
                )
            if not self._is_SequentialPrivacyFrame():
                raise AttributeError(
                    "Some attributes where not secified at creation, columns are:"
                    + str(self.columns)
                    + " but ''(%s, %s, %s, %s, %s)'' are needed"
                    % (
                        constants.USER_ID,
                        constants.ORDER_ID,
                        constants.SEQUENCE_ID,
                        constants.DATETIME,
                        constants.ELEMENTS,
                    )
                )

    def _check_order_date(self):
        if constants.SEQUENCE_ID not in self:
            self[constants.SEQUENCE_ID] = self[constants.USER_ID]
        if constants.DATETIME in self and constants.ORDER_ID in self:
            sort4sequence = self.sort_values(
                by=[constants.USER_ID, constants.SEQUENCE_ID, constants.ORDER_ID]
            )
            sort4date = self.sort_values(by=[constants.USER_ID, constants.DATETIME])
            if not sort4sequence.equals(sort4date):
                raise AttributeError(
                    f"PrivacyDataFrame {constants.DATETIME} attribute doesn't match with {constants.ORDER_ID}"
                )

        if constants.DATETIME in self and constants.ORDER_ID not in self:
            self.sort_values(
                by=[constants.USER_ID, constants.SEQUENCE_ID, constants.DATETIME],
                inplace=True,
            )
            self[constants.ORDER_ID] = 0
            self._make_order()

        if constants.DATETIME not in self and constants.ORDER_ID not in self:
            self.sort_values(
                by=[constants.USER_ID, constants.SEQUENCE_ID], inplace=True
            )
            self[constants.ORDER_ID] = 0
            self._make_order()

        if (
            constants.DATETIME in self
            and not pd.core.dtypes.common.is_datetime64_any_dtype(
                self[constants.DATETIME].dtype
            )
        ):
            self[constants.DATETIME] = pd.to_datetime(self[constants.DATETIME])

    def _make_order(self):
        df_iterator = self.iterrows()

        index, row = next(df_iterator)
        self.loc[index, constants.ORDER_ID] = 1
        puid, pseq = row[constants.USER_ID], row[constants.SEQUENCE_ID]

        val = 2
        for index, row in df_iterator:
            if (row[constants.USER_ID] == puid) and (
                row[constants.SEQUENCE_ID] == pseq
            ):
                self.loc[index, constants.ORDER_ID] = val
            else:
                val = 1
                self.loc[index, constants.ORDER_ID] = 1
                puid, pseq = row[constants.USER_ID], row[constants.SEQUENCE_ID]
            val += 1

    def _is_SequentialPrivacyFrame(self):
        return (
            (constants.USER_ID in self)
            and (constants.ORDER_ID in self)
            and (constants.SEQUENCE_ID in self)
            and (constants.DATETIME in self)
            and (constants.ELEMENTS in self)
        )

    @property
    def uid(self):
        if constants.USER_ID not in self:
            raise AttributeError(
                "The PrivacyDataFrame does not contain the column '%s.'"
                % constants.USER_ID
            )
        return self[constants.USER_ID]

    @property
    def order(self):
        if constants.ORDER_ID not in self:
            raise AttributeError(
                "The PrivacyDataFrame does not contain the column '%s.'"
                % constants.ORDER_ID
            )
        return self[constants.ORDER_ID]

    @property
    def datetime(self):
        if constants.DATETIME not in self:
            raise AttributeError(
                "The PrivacyDataFrame does not contain the column '%s.'"
                % constants.DATETIME
            )
        return self[constants.DATETIME]

    @property
    def sequence(self):
        if constants.SEQUENCE_ID not in self:
            raise AttributeError(
                "The PrivacyDataFrame does not contain the column '%s.'"
                % constants.SEQUENCE_ID
            )
        return self[constants.SEQUENCE_ID]

    @property
    def elements(self):
        if constants.ELEMENTS not in self:
            raise AttributeError(
                "The PrivacyDataFrame does not contain the column '%s.'"
                % constants.ELEMENTS
            )
        return self[constants.ELEMENTS]

    @classmethod
    def from_file(
        cls,
        filename,
        user_id=constants.USER_ID,
        datetime=constants.DATETIME,
        order_id=constants.ORDER_ID,
        sequence_id=constants.SEQUENCE_ID,
        elements=constants.ELEMENTS,
        timestamp=False,
        check_order_date=True,
        column_type=None,
        encoding=None,
        usecols=None,
        header="infer",
        sep=",",
    ):
        df = pd.read_csv(
            filename, sep=sep, header=header, usecols=usecols, encoding=encoding
        )

        return cls(
            df,
            user_id=user_id,
            datetime=datetime,
            order_id=order_id,
            sequence_id=sequence_id,
            elements=elements,
            timestamp=timestamp,
            check_order_date=check_order_date,
        )

    @property
    def _constructor_sliced(self):
        return SequentialSeries

    @property
    def _constructor(self):
        return SequentialPrivacyFrame
