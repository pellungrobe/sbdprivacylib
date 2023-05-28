from abc import ABC, abstractmethod
from .utils import date_time_precision
from . import constants
from pandas.errors import AbstractMethodError
import pandas as pd

__all__ = [
    "ElementsAttack",
    "SequenceAttack",
    "TimeAttack",
    "FrequencyAttack",
    "ProbabilityAttack",
    "ProportionAttack",
]


class BackgroundKnowledgeAttack(ABC):
    """Privacy Attack

    Abstract class for a generic background knowledge based attack. Defines a series of functions common to all attacks.
    Provides basic functions to match a background knowledge instance to individual's data and a preprocessing function.

    """

    @abstractmethod
    def preprocess(data, **kwargs):
        """preprocess

        Function to perform preprocess of the data. Can be dependant on the RiskEvaluator by using the aggregation_levels
        function. Abstract method, all BackgroundKnowledgeAttacks must implement it.

        Parameters
        ----------
        data : SequentialPrivacyFrame
            the entire data to be preprocessed before attack simulation.

        **kwargs : mapping, optional
            further arguments for preprocessing that can be passed from the RiskEvaluator, for example aggregation_levels
        """
        raise AbstractMethodError(data)

    @abstractmethod
    def matching(single_priv_df, case):
        """matching

        Function to perform matching between a background knowledge instance and the data of an individual

        Parameters
        ----------
        single_priv_df : SequentialPrivacyFrame
            the data of a single individual.

        case : list or numpy array or dict
            the background knowledge instance.
        """
        raise AbstractMethodError(single_priv_df)


class TabularAttack:
    """TabularAttack

    A Tabular Attack for Tabular data.

    Parameters
    ----------
    data : DataFrame
        the data on which to perform privacy risk assessment simulating this attack.

    **kwargs : mapping, optional
        a dictionary of keyword arguments passed into the preprocessing of attack.

    """

    def matching(single_priv_df, case, tolerance):
        """matching
        Matching function for the attack.
        For TabularAttack, it checks all the values in the row under examination.

        Parameters
        ----------
        single_priv_df : DataFrame
            the data of a single individual.

        case : tuple of tuples
            the background knowledge instance.
        Returns
        -------
        int
            1 if the instance matches the single_priv_df, 0 otherwise.
        """
        for n, v in case:
            bar = single_priv_df[n].values[0]
            if type(v) == "int" or type(v) == "float":
                if not (v < bar + bar * tolerance and v > bar - bar * tolerance):
                    return 0
            elif bar != v:
                return 0
        return 1


class ElementsAttack(BackgroundKnowledgeAttack):
    """ElementsAttack

    In an ElementsAttack the adversary knows some elements in the sequences of an individual.

    Parameters
    ----------
    data : SequentialPrivacyFrame
        the data on which to perform privacy risk assessment simulating this attack.

    **kwargs : mapping, optional
        a dictionary of keyword arguments passed into the preprocessing of attack.

    """

    def preprocess(data, **kwargs):
        """preprocess

        Function to perform preprocess of the data.

        Parameters
        ----------
        data : SequenctialPrivacyFrame
            the entire data to be preprocessed before attack simulation.

        **kwargs : mapping, optional
            further arguments for preprocessing that can be passed from the RiskEvaluator, for example aggregation_levels
        """
        data.sort_values(
            by=[constants.USER_ID, constants.DATETIME], ascending=True, inplace=True
        )
        return data

    def matching(single_priv_df, case):
        """matching
        Matching function for the attack.
        For ElementsAttack, only the elements are used in the matching.

        Parameters
        ----------
        single_priv_df : SequentialPrivacyFrame
            the data of a single individual.

        case : list or numpy array or dict
            the background knowledge instance.
        Returns
        -------
        int
            1 if the instance matches the single_priv_df, 0 otherwise.
        """
        occ = pd.DataFrame(data=case, columns=single_priv_df.columns)
        occ = occ.astype(dtype=dict(single_priv_df.dtypes))
        occ = (
            occ.groupby([constants.ELEMENTS])
            .size()
            .reset_index(name=constants.COUNT + "case")
        )
        single_priv_grouped = (
            single_priv_df.groupby([constants.ELEMENTS])
            .size()
            .reset_index(name=constants.COUNT)
        )
        merged = pd.merge(
            single_priv_grouped,
            occ,
            left_on=[constants.ELEMENTS],
            right_on=[constants.ELEMENTS],
        )
        if len(merged.index) != len(occ.index):
            return 0
        else:
            condition = merged[constants.COUNT] >= merged[constants.COUNT + "case"]
            if len(merged[condition].index) != len(occ.index):
                return 0
            else:
                return 1


class SequenceAttack(BackgroundKnowledgeAttack):
    """SequenceAttack

    In an SequenceAttack the adversary knows some elements in the sequences of an individual and the orders with which they
    appear.

    Parameters
    ----------
    data : SequentialPrivacyFrame
        the data on which to perform privacy risk assessment simulating this attack.

    **kwargs : mapping, optional
        a dictionary of keyword arguments passed into the preprocessing of attack.

    """

    def preprocess(data, **kwargs):
        """preprocess

        Function to perform preprocess of the data.

        Parameters
        ----------
        data : SequentialPrivacyFrame
            the entire data to be preprocessed before attack simulation.

        **kwargs : mapping, optional
            further arguments for preprocessing that can be passed from the RiskEvaluator, for example aggregation_levels
        """
        data.sort_values(
            by=[constants.USER_ID, constants.ORDER_ID], ascending=True, inplace=True
        )
        return data

    def matching(single_priv_df, case):
        """matching
        Matching function for the attack.
        For SequenceAttack, elements and their relative order are used in the matching.

        Parameters
        ----------
        single_priv_df : SequentialPrivacyFrame
            the data of a single individual.

        case : list or numpy array or dict
            the background knowledge instance.
        Returns
        -------
        int
            1 if the instance matches the single_priv_df, 0 otherwise.
        """
        occ = pd.DataFrame(data=case, columns=single_priv_df.columns)
        occ_iterator = occ.iterrows()
        occ_line = next(occ_iterator)[1]

        count = 0
        for index, row in single_priv_df.iterrows():
            if row[constants.ELEMENTS] == occ_line[constants.ELEMENTS]:
                count += 1
                try:
                    occ_line = next(occ_iterator)[1]
                except StopIteration:
                    break
        if len(occ.index) == count:
            return 1
        else:
            return 0


class TimeAttack(BackgroundKnowledgeAttack):
    """TimeAttack

    In an TimeAttack the adversary knows some elements in the sequences of an individual and the datetime with which they
    appear.

    Parameters
    ----------
    data : SequentialPrivacyFrame
        the data on which to perform privacy risk assessment simulating this attack.

    **kwargs : mapping, optional
        a dictionary of keyword arguments passed into the preprocessing of attack.

    """

    def preprocess(data, **kwargs):
        """preprocess

        Function to perform preprocess of the data.

        Parameters
        ----------
        data : SequentialPrivacyFrame
            the entire data to be preprocessed before attack simulation.

        **kwargs : mapping, optional
            further arguments for preprocessing that can be passed from the RiskEvaluator, for example aggregation_levels
        """
        if constants.DATETIME not in data:
            raise AttributeError(
                f"PrivacyDataFrame doesn't contain attribute {constants.DATETIME}"
            )
        precision = constants.HOUR
        if constants.PRECISION in kwargs:
            precision = kwargs[constants.PRECISION]
        if precision not in constants.PRECISION_LEVELS:
            raise AttributeError(f"Precision values unrecognized {constants.DATETIME}")
        data[constants.TEMP] = data[constants.DATETIME].apply(
            lambda x: date_time_precision(x, precision)
        )
        data.sort_values(
            by=[constants.USER_ID, constants.DATETIME], ascending=True, inplace=True
        )
        return data

    def matching(single_priv_df, case):
        """matching
        Matching function for the attack.
        For TimeAttack, elements and their datetime are used in the matching.

        Parameters
        ----------
        single_priv_df : SequentialPrivacyFrame
            the data of a single individual.

        case : list or numpy array or dict
            the background knowledge instance.
        Returns
        -------
        int
            1 if the instance matches the single_priv_df, 0 otherwise.
        """
        occ = pd.DataFrame(data=case, columns=single_priv_df.columns)
        occ = (
            occ.groupby([constants.ELEMENTS, constants.TEMP])
            .size()
            .reset_index(name=constants.COUNT + "case")
        )
        single_priv_grouped = (
            single_priv_df.groupby([constants.ELEMENTS, constants.TEMP])
            .size()
            .reset_index(name=constants.COUNT)
        )
        merged = pd.merge(
            single_priv_grouped,
            occ,
            left_on=[constants.ELEMENTS, constants.TEMP],
            right_on=[constants.ELEMENTS, constants.TEMP],
        )

        if len(merged.index) != len(occ.index):
            return 0
        else:
            cond = merged[constants.COUNT] >= merged[constants.COUNT + "case"]
            if len(merged[cond].index) != len(occ.index):
                return 0
            else:
                return 1


class FrequencyAttack(BackgroundKnowledgeAttack):
    """FrequencyAttack

    In an FrequencyAttack the adversary knows some elements in the sequences of an individual and the frequency with which they
    appear.

    Parameters
    ----------
    data : SequentialPrivacyFrame
        the data on which to perform privacy risk assessment simulating this attack.

    **kwargs : mapping, optional
        a dictionary of keyword arguments passed into the preprocessing of attack.

    """

    def preprocess(data, **kwargs):
        """preprocess

        Function to perform preprocess of the data.

        Parameters
        ----------
        data : SequentialPrivacyFrame
            the entire data to be preprocessed before attack simulation.

        **kwargs : mapping, optional
            further arguments for preprocessing that can be passed from the RiskEvaluator, for example aggregation_levels
        """
        aggregation_levels = kwargs["aggregation_levels"]
        tolerance = 0.0
        if constants.TOLERANCE in kwargs:
            tolerance = kwargs[constants.TOLERANCE]
        frequency_vector = (
            data.groupby(aggregation_levels)
            .size()
            .reset_index(name=constants.FREQUENCY)
        )
        frequency_vector[constants.TOLERANCE] = tolerance
        return frequency_vector.sort_values(by=[constants.USER_ID, constants.FREQUENCY])

    def matching(single_priv_df, case):
        """matching
        Matching function for the attack.
        For FrequencyAttack, elements and their frequency are used in the matching.

        Parameters
        ----------
        single_priv_df : SequentialPrivacyFrame
            the data of a single individual.

        case : list or numpy array or dict
            the background knowledge instance.
        Returns
        -------
        int
            1 if the instance matches the single_priv_df, 0 otherwise.
        """
        occ = pd.DataFrame(data=case, columns=single_priv_df.columns)
        occ.rename(
            columns={
                constants.FREQUENCY: constants.FREQUENCY + "case",
                constants.TOLERANCE: constants.TOLERANCE + "case",
            },
            inplace=True,
        )
        merged = pd.merge(
            single_priv_df,
            occ,
            left_on=[constants.ELEMENTS],
            right_on=[constants.ELEMENTS],
        )

        if len(merged.index) != len(occ.index):
            return 0
        else:
            cond1 = merged[constants.FREQUENCY + "case"] >= merged[
                constants.FREQUENCY
            ] - (merged[constants.FREQUENCY] * merged[constants.TOLERANCE])
            cond2 = merged[constants.FREQUENCY + "case"] <= merged[
                constants.FREQUENCY
            ] + (merged[constants.FREQUENCY] * merged[constants.TOLERANCE])
            if len(merged[cond1 & cond2].index) != len(occ.index):
                return 0
            else:
                return 1


class ProbabilityAttack(BackgroundKnowledgeAttack):
    """ProbabilityAttack

    In an FrequencyAttack the adversary knows some elements in the sequences of an individual and the probability with which they
    appear.

    Parameters
    ----------
    data : SequentialPrivacyFrame
        the data on which to perform privacy risk assessment simulating this attack.

    **kwargs : mapping, optional
        a dictionary of keyword arguments passed into the preprocessing of attack.

    """

    def preprocess(data, **kwargs):
        """preprocess

        Function to perform preprocess of the data.

        Parameters
        ----------
        data : SequentialPrivacyFrame
            the entire data to be preprocessed before attack simulation.

        **kwargs : mapping, optional
            further arguments for preprocessing that can be passed from the RiskEvaluator, for example aggregation_levels
        """
        aggregation_levels = kwargs["aggregation_levels"]
        tolerance = 0.0
        if constants.TOLERANCE in kwargs:
            tolerance = kwargs[constants.TOLERANCE]
        num = (
            data.groupby(aggregation_levels)
            .size()
            .reset_index(name=constants.FREQUENCY)
        )
        dim = (
            data.groupby(aggregation_levels[:-1])
            .size()
            .reset_index(name=constants.TOTAL_FREQUENCY)
        )
        probability_vector = pd.merge(
            num, dim, left_on=aggregation_levels[:-1], right_on=aggregation_levels[:-1]
        )
        probability_vector[constants.PROBABILITY] = (
            probability_vector[constants.FREQUENCY]
            / probability_vector[constants.TOTAL_FREQUENCY]
        )
        probability_vector[constants.TOLERANCE] = tolerance
        return probability_vector.sort_values(
            by=[constants.USER_ID, constants.PROBABILITY]
        )

    def matching(single_priv_df, case):
        """matching
        Matching function for the attack.
        For ProbabilityAttack, elements and their probability are used in the matching.

        Parameters
        ----------
        single_priv_df : SequentialPrivacyFrame
            the data of a single individual.

        case : list or numpy array or dict
            the background knowledge instance.
        Returns
        -------
        int
            1 if the instance matches the single_priv_df, 0 otherwise.
        """
        occ = pd.DataFrame(data=case, columns=single_priv_df.columns)
        occ.rename(
            columns={
                constants.PROBABILITY: constants.PROBABILITY + "case",
                constants.TOLERANCE: constants.TOLERANCE + "case",
            },
            inplace=True,
        )
        merged = pd.merge(
            single_priv_df,
            occ,
            left_on=[constants.ELEMENTS],
            right_on=[constants.ELEMENTS],
        )

        if len(merged.index) != len(occ.index):
            return 0
        else:
            cond1 = merged[constants.PROBABILITY + "case"] >= merged[
                constants.PROBABILITY
            ] - (merged[constants.PROBABILITY] * merged[constants.TOLERANCE])
            cond2 = merged[constants.PROBABILITY + "case"] <= merged[
                constants.PROBABILITY
            ] + (merged[constants.PROBABILITY] * merged[constants.TOLERANCE])
            if len(merged[cond1 & cond2].index) != len(merged.index):
                return 0
            else:
                return 1


class ProportionAttack(BackgroundKnowledgeAttack):
    """ProportionAttack

    In an ProportionAttack the adversary knows some elements in the sequences of an individual and the proportion
    with which they appear w.r.t. the most frequent elements in the sequences.

    Parameters
    ----------
    data : SequentialPrivacyFrame
        the data on which to perform privacy risk assessment simulating this attack.

    **kwargs : mapping, optional
        a dictionary of keyword arguments passed into the preprocessing of attack.

    """

    def preprocess(data, **kwargs):
        """preprocess

        Function to perform preprocess of the data.

        Parameters
        ----------
        data : SequentialPrivacyFrame
            the entire data to be preprocessed before attack simulation.

        **kwargs : mapping, optional
            further arguments for preprocessing that can be passed from the RiskEvaluator, for example aggregation_levels
        """
        aggregation_levels = kwargs["aggregation_levels"]
        tolerance = 0.0
        if constants.TOLERANCE in kwargs:
            tolerance = kwargs[constants.TOLERANCE]
        frequency_vector = (
            data.groupby(aggregation_levels)
            .size()
            .reset_index(name=constants.FREQUENCY)
        )
        frequency_vector[constants.TOLERANCE] = tolerance
        return frequency_vector.sort_values(by=[constants.USER_ID, constants.FREQUENCY])

    def matching(single_priv_df, case):
        """matching
        Matching function for the attack.
        For ProportionAttack, elements and their proportion w.r.t. the most frequent element are used in the matching.

        Parameters
        ----------
        single_priv_df : SequentialPrivacyFrame
            the data of a single individual.

        case : list or numpy array or dict
            the background knowledge instance.
        Returns
        -------
        int
            1 if the instance matches the single_priv_df, 0 otherwise.
        """
        occ = pd.DataFrame(data=case, columns=single_priv_df.columns)
        occ.rename(
            columns={
                constants.FREQUENCY: constants.FREQUENCY + "case",
                constants.TOLERANCE: constants.TOLERANCE + "case",
            },
            inplace=True,
        )
        merged = pd.merge(
            single_priv_df,
            occ,
            left_on=[constants.ELEMENTS],
            right_on=[constants.ELEMENTS],
        )

        if len(merged.index) != len(occ.index):
            return 0
        else:
            merged[constants.PROPORTION + "case"] = (
                merged[constants.FREQUENCY + "case"]
                / merged[constants.FREQUENCY + "case"].max()
            )
            merged[constants.PROPORTION] = (
                merged[constants.FREQUENCY] / merged[constants.FREQUENCY].max()
            )
            cond1 = merged[constants.PROPORTION + "case"] >= merged[
                constants.PROPORTION
            ] - (merged[constants.PROPORTION] * merged[constants.TOLERANCE])
            cond2 = merged[constants.PROPORTION + "case"] <= merged[
                constants.PROPORTION
            ] + (merged[constants.PROPORTION] * merged[constants.TOLERANCE])
            if len(merged[cond1 & cond2].index) != len(occ.index):
                return 0
            else:
                return 1
