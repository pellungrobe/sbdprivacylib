from abc import ABC, abstractmethod
from itertools import combinations, chain
from sys import maxsize

from pandas.errors import AbstractMethodError
import pandas as pd

from . import constants
from .attacks import BackgroundKnowledgeAttack
from .sequentialprivacyframe import SequentialPrivacyFrame

__all__ = ["IndividualElementEvaluator", "IndividualSequenceEvaluator"]

class SequentialRiskEvaluator(ABC):
    """SequentialRiskEvaluator

    Abstract class for a generic risk evaluator. It implements methods and function useful for any background knowledge
    based attack simulation.

    Parameters
    ----------
    data : SequentialPrivacyFrame
        the data on which to perform privacy risk assessment.

    attack : BackgroundKnowledgeAttack
        an attack to be simulated. Must be a class implementing the BackgroundKnowledgeAttack abstract class

    knowledge_length : int
        the length of the knowledge of the simultated attack, i.e., how many data points are assumed to be in the
        background knowledge of the adversary

    **kwargs : mapping, optional
        a dictionary of keyword arguments passed into the preprocessing of attack.

    References
    ----------
    .. [TIST2018] Roberto Pellungrini, Luca Pappalardo, Francesca Pratesi, and Anna Monreale. 2017. A Data Mining Approach to Assess Privacy Risk in Human Mobility Data. ACM Trans. Intell. Syst. Technol. 9, 3, Article 31 (December 2017), 27 pages. DOI: https://doi.org/10.1145/3106774
    .. [MOB2018] Roberto Pellungrini, Luca Pappalardo, Francesca Pratesi, Anna Monreale: Analyzing Privacy Risk in Human Mobility Data. STAF Workshops 2018: 114-129
    """
    
    def __init__(self, data, attack, knowledge_length, **kwargs):
        if not isinstance(data, SequentialPrivacyFrame):
            raise AttributeError(f"Evaluators must process SequentialPrivacyFrame in input, but data of type {type(data)} was passed.")
        if not issubclass(attack, BackgroundKnowledgeAttack):
            raise AttributeError(f"Attacks must implement the BackgroundKnowledgeAttack class, type {type(attack)} was passed.")
        if not isinstance(knowledge_length, int):
            raise AttributeError(f"Knowledge lenght must be a integer value, type {type(knowledge_length)} was passed.")
        self._attack = attack
        if knowledge_length <= 0:
            self._knowledge_length = maxsize
        else:
            self._knowledge_length = knowledge_length
        self._data = self.attack.preprocess(data, aggregation_levels=self.aggregation_levels(), **kwargs)
        
    @property
    def data(self):
        return self._data
    
    @property
    def attack(self):
        return self._attack 
    
    @property
    def knowledge_length(self):
        return self._knowledge_length
        
    @knowledge_length.setter
    def knowledge_length(self, knowledge_length):
        if not isinstance(knowledge_length, int):
            raise AttributeError(f"Knowledge lenght must be a integer value, type {type(knowledge_length)} was passed.")
        self._knowledge_length = knowledge_length
        
    @abstractmethod
    def background_knowledge_gen(self, single_privacy_frame):
        """background_knowledge_gen

        Generates all possible combinations of knowledge_length length with respect to the current evaluator.
        Abstract method, all RiskEvaluators must implement it.

        Parameters
        ----------
        single_priv_df : SequentialPrivacyFrame
            the data of the single individual from which to generate all possible background knowledge instances.
        """
        raise AbstractMethodError(self)
    
    @abstractmethod
    def __risk(self, single_privacy_frame):
        """risk

        Computes the privacy risk for a single individual. Abstract method, all RiskEvaluators must implement it.

        Parameters
        ----------
        single_priv_df : SequentialPrivacyFrame
            the data of the single individual from which to generate all possible background knowledge instances.

        """
        raise AbstractMethodError(self)
    
    @abstractmethod
    def aggregation_levels(self):
        """aggregation_levels

        Allows attack preprocess to be dependant on the logic of the RiskEvaluator if needed. Data can be
        preprocessed at RiskEvaluator creation by the AttackModel using aggregation_levels to determine which
        attributes in the sequence needs to be grouped. Abstract method, all RiskEvaluators must implement it.

        """
        raise AbstractMethodError(self)
    
    def assess_risk(self, targets=None):
        """assess_risk

            Assesses privacy risk for the data fed to this evaluator, using the attack specified at evaluator construction.

            Parameters
            ----------
            targets : TrajectoryDataFrame or list, optional
                the users_id target of the attack.  They must be compatible with the sequence data. If None is used,
                risk is computed on all users in traj. The default is `None`.

            Returns
            -------
            risks : Dataframe
                a dataframe in the form (user id, privacy risk).
            """
        if targets is None:
            targets = self.data
        elif isinstance(targets, list):
            targets = self.data[self.data[constants.USER_ID].isin(targets)]
        elif isinstance(targets, SequentialPrivacyFrame) or isinstance(targets, pd.SequentialPrivacyFrame):
            targets = self.data[self.data[constants.USER_ID].isin(targets[constants.USER_ID])]
        else:
            raise AttributeError("Targets must be either a list of user_ids or a dataframe. Leave empty for total dataset assessment")
        risks = targets.groupby(constants.USER_ID).apply(lambda x: self.__risk(x)).reset_index(
                name=constants.PRIVACY_RISK)
        return risks
    
class IndividualElementEvaluator(SequentialRiskEvaluator):
    """IndividualElementEvaluator

    Class for evaluating risk on individual level: risk is computed based on the whole data of each individual, i.e., each
    individual risk will be equal to the inverse of the number of other individuals in the data that match the background knowledge.

    Parameters
    ----------
    data : SequentialPrivacyFrame
        the data on which to perform privacy risk assessment.

    attack : BackgroundKnowledgeAttack
        an attack to be simulated. Must be a class implementing the BackgroundKnowledgeAttack abstract class

    knowledge_length : int
        the length of the knowledge of the simultated attack, i.e., how many data points are assumed to be in the
        background knowledge of the adversary

    **kwargs : mapping, optional
        a dictionary of keyword arguments passed into the preprocessing of attack.

    References
    ----------
    .. [TIST2018] Roberto Pellungrini, Luca Pappalardo, Francesca Pratesi, and Anna Monreale. 2017. A Data Mining Approach to Assess Privacy Risk in Human Mobility Data. ACM Trans. Intell. Syst. Technol. 9, 3, Article 31 (December 2017), 27 pages. DOI: https://doi.org/10.1145/3106774
    .. [MOB2018] Roberto Pellungrini, Luca Pappalardo, Francesca Pratesi, Anna Monreale: Analyzing Privacy Risk in Human Mobility Data. STAF Workshops 2018: 114-129
    """

    def background_knowledge_gen(self, single_priv_df):
        """background_knowledge_gen

            Generates all possible combinations of knowledge_length length from the data of an individual, to provide all
            possible background knowledge instances to the simulation.

            Parameters
            ----------
            single_priv_df : SequentialPrivacyFrame
                the data of the single individual from which to generate all possible background knowledge instances.

            Returns
            -------
            cases : iterator
                an iterator over all possible combinations of data points, i.e., all possible background knowledge instances.
            """
        size = len(single_priv_df)
        if self.knowledge_length > size:
            cases = combinations(single_priv_df.values, size)
        else:
            cases = combinations(single_priv_df.values, self.knowledge_length)
        return cases
    
    def __risk(self, single_privacy_frame):
        """risk

        Computes the privacy risk for a single individual

        Parameters
        ----------
        single_priv_df : SequentialPrivacyFrame
            the data of the single individual from which to generate all possible background knowledge instances.

        Returns
        -------
        privacy_risk : float
            the privacy risk for the individual, computed as the inverse of the number of other individuals in the data
            that match the background knowledge.
        """
        cases = self.background_knowledge_gen(single_privacy_frame)
        privacy_risk = 0
        for case in cases:
            case_risk = 1.0 / self.data.groupby(constants.USER_ID).apply(lambda x: self.attack.matching(x, case)).sum()
            if case_risk > privacy_risk:
                privacy_risk = case_risk
            if privacy_risk == 1:
                break
        return privacy_risk
    
    def aggregation_levels(self):
        """aggregation_levels

        Allows attack preprocess to be dependant on the logic of the RiskEvaluator if needed.
        For IndividualElementEvaluator, aggregation is done for each individual in the data and for each distinct element
        belonging to the individual.

        Returns
        -------
        list
            a list with the attributes to be aggregated, should an attack need it. For IndividualElementEvaluator these
            are user id and the elements of the sequence.
        """
        return [constants.USER_ID,constants.ELEMENTS]

class IndividualSequenceEvaluator(IndividualElementEvaluator):
    """IndividualSequenceEvaluator

    Class for evaluating risk on sequence level: risk is computed based on the different sequences of each individual, i.e., each
    individual risk will be equal to the number of sequences in her own data divided by the total number of sequences
    belonging to other individuals in the data that match the background knowledge.

    Parameters
    ----------
    data : SequentialPrivacyFrame
        the data on which to perform privacy risk assessment.

    attack : BackgroundKnowledgeAttack
        an attack to be simulated. Must be a class implementing the BackgroundKnowledgeAttack abstract class

    knowledge_length : int
        the length of the knowledge of the simultated attack, i.e., how many data points are assumed to be in the
        background knowledge of the adversary

    **kwargs : mapping, optional
        a dictionary of keyword arguments passed into the preprocessing of attack.

    References
    ----------
    .. [TIST2018] Roberto Pellungrini, Luca Pappalardo, Francesca Pratesi, and Anna Monreale. 2017. A Data Mining Approach to Assess Privacy Risk in Human Mobility Data. ACM Trans. Intell. Syst. Technol. 9, 3, Article 31 (December 2017), 27 pages. DOI: https://doi.org/10.1145/3106774
    .. [MOB2018] Roberto Pellungrini, Luca Pappalardo, Francesca Pratesi, Anna Monreale: Analyzing Privacy Risk in Human Mobility Data. STAF Workshops 2018: 114-129
    """
    
    def background_knowledge_gen(self, single_priv_df):
        """background_knowledge_gen

            Generates all possible combinations of knowledge_length length from the data of an individual, to provide all
            possible background knowledge instances to the simulation.

            Parameters
            ----------
            single_priv_df : SequentialPrivacyFrame
                the data of the single individual from which to generate all possible background knowledge instances.

            Returns
            -------
            cases : iterator
            an iterator over all possible combinations of data points, i.e., all possible background knowledge instances.
            """
        cases = chain(*single_priv_df.groupby(constants.SEQUENCE_ID).apply(lambda x: super().background_knowledge_gen(x)))
        return cases
     
    def __risk(self, single_privacy_frame):
        """risk

        Computes the privacy risk for a single individual

        Parameters
        ----------
        single_priv_df : SequentialPrivacyFrame
            the data of the single individual from which to generate all possible background knowledge instances.

        Returns
        -------
        privacy_risk : float
            the privacy risk for the individual, computed as the number of sequences belonging to the individual divided
            by the number of all sequences in the data that match the brackground knowledge.
        """
        cases = self.background_knowledge_gen(single_privacy_frame)
        privacy_risk = 0
        for case in cases:
            num = single_privacy_frame.groupby([constants.SEQUENCE_ID]).apply(lambda x: self.attack.matching(x, case)).sum()
            den = self.data.groupby([constants.USER_ID, constants.SEQUENCE_ID]).apply(lambda x: self.attack.matching(x, case)).sum()
            case_risk = num / den

            if case_risk > privacy_risk:
                privacy_risk = case_risk

            if privacy_risk == 1:
                break
        return privacy_risk
    
    def aggregation_levels(self):
        """aggregation_levels

        Allows attack preprocess to be dependant on the logic of the RiskEvaluator if needed.
        For IndividualSequenceEvaluator, aggregation is done for each individual in the data and for each sequence and distinct
        element that belong to the individual.

        Returns
        -------
        list
            a list with the attributes to be aggregated, should an attack need it. For IndividualSequenceEvaluator these
            are user id, sequence id and the elements of the sequence.
        """
        return [constants.USER_ID,constants.SEQUENCE_ID,constants.ELEMENTS]
