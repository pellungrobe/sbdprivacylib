from privlib.anonymization.src.attribute_types.value import Value
from privlib.anonymization.src.utils import constants
import nltk
from nltk.corpus import wordnet
from random import sample


class Semantic_categorical_wordnet(Value):
    """Semantic_categorical_wordnet

    Class that implements the necessary methods to deal with attribute type Semantic categorical in wordnet

    """

    reference_value = None
    sims_wp = {}
    try:
        nltk.find("corpora/wordnet")
    except LookupError:
        print("Downloading wordnet (only the first time)")
        nltk.download("wordnet")

    def __init__(self, value):
        """Constructor, called from inherited classes
        Creates an instance of the attribute type for Semantic categorical value

        Parameters
        ----------
        value :
            the Semantic categorical is received as an string

        See Also
        --------
        :class:`Value`
        """
        syns = wordnet.synsets(value, pos="n")
        if len(syns) == 0:
            raise TypeError(f"Value not found in Wordnet : {value}")
        self.value = value
        self.syn = None
        self.id = 0

    def distance(self, value):
        """distance

        Calculates the distance between this Semantic categorical and the received value

        Parameters
        ----------
        value :
            The other Semantic categorical value to calculate the distance.

        Returns
        -------
        float
            The distance between the two Semantic categorical values.

        See Also
        --------
        :class:`Value`
        """
        dist = 1 - Semantic_categorical_wordnet.similarity_wp(self.value, value.value)

        return dist

    @staticmethod
    def similarity_wp(word1, word2):
        """similarity_wp

        Calculates the similarity in wordnet ontology between this semantic categorical and the received value

        Parameters
        ----------
        word1 :
            The semantic categorical value to calculate the distance.

        word2 :
            The other semantic categorical value to calculate the distance.

        Returns
        -------
        float
            The similarity between the two semantic categorical values in wordnet ontology.
            The values are disambiguated as the synsets with the maximum similarity in wordnet
        """
        key = word1 + "," + word2
        if key in Semantic_categorical_wordnet.sims_wp:
            return Semantic_categorical_wordnet.sims_wp[key]
        else:
            key = word2 + "," + word1
            if key in Semantic_categorical_wordnet.sims_wp:
                return Semantic_categorical_wordnet.sims_wp[key]

        syns1 = wordnet.synsets(word1, pos="n")
        syns2 = wordnet.synsets(word2, pos="n")
        max_sim = 0
        for syn1 in syns1:
            for syn2 in syns2:
                sim = syn1.wup_similarity(syn2)
                if sim > max_sim:
                    max_sim = sim
        Semantic_categorical_wordnet.sims_wp[key] = max_sim

        return max_sim

    @staticmethod
    def similarity_wp_word_syn(word, syn):
        """similarity_wp_word_syn

        The similarity between the two semantic categorical values in wordnet ontology.
        The values are disambiguated given the number of wordnet synset as parameter

        Parameters
        ----------
        word :
            The other semantic categorical value to calculate the distance.

        syn : int
            The number of synset of the other semantic categorical value to calculate the distance.

        Returns
        -------
        float
            The similarity between the two semantic categorical values.
        """
        syns_word = wordnet.synsets(word, pos="n")
        max_sim = 0
        for syn_word in syns_word:
            sim = syn_word.wup_similarity(syn)
            if sim > max_sim:
                max_sim = sim

        return max_sim

    @staticmethod
    def mean_similarity_wp(syn, list_syns):
        """mean_similarity_wp

        Calculates the mean similarity of a synset and a list of synsets in wordnet ontology.

        Parameters
        ----------
        syn :
            The synset in wordnet to calculate the mean similarity.

        list_syns :
            The list of synsets in wordnet to calculate the mean similarity.

        Returns
        -------
        float
            The mean similarity in wordnet of the synset w.r.t. the list of synsets.
        """
        mean = 0
        for syn2 in list_syns:
            mean += syn.wup_similarity(syn2)
        mean /= len(list_syns)

        return mean

    @staticmethod
    def calculate_centroid(values, **kwargs):
        """calculate_centroid

        Calculates the semantic categorical value that is the centroid of the list of semantic categorical values
        given as parameter. The centroid is the value in the list of candidates that maximizes the similarity
        to the semantic categorical values in the given list of values.
        The list of candidates is generated as the set of all parents in wordnet of values in the list.

        Parameters
        ----------
        values :
            The list of semantic categorical values to calculate its centroid

        **kwargs : optional
            Additional arguments that the specific attribute type value may need to calculate the centroid

        Returns
        -------
        Semantic_categorical_wordnet
            The value that is the centroid of the list of semantic categorical values.
        """
        if constants.EPSILON in kwargs.keys():
            centroid = Semantic_categorical_wordnet.calculate_dp_centroid(
                values, **kwargs
            )
            return centroid
        # centroid that maximizes similarity
        candidates = set()
        for value in values:
            for syn in wordnet.synsets(value.value, pos="n"):
                for parents in syn.hypernym_paths():
                    for parent in parents:
                        candidates.add(parent)
        syn_max = sample(candidates, 1)[0]
        max_sim = Semantic_categorical_wordnet.mean_similarity_wp(syn_max, candidates)
        for syn in candidates:
            sim = Semantic_categorical_wordnet.mean_similarity_wp(syn, candidates)
            if sim > max_sim:
                max_sim = sim
                syn_max = syn
        centroid = Semantic_categorical_wordnet(syn_max.lemmas()[0].name())
        centroid.syn = syn_max

        return centroid

    @staticmethod
    def calculate_dp_centroid(values, **kwargs):
        """calculate_centroid

        Calculates the semantic categorical value that is the differential private centroid of
        the list of semantic categorical values given as parameter
        In this case, the centroid is the same as the semantic categorical centroid.

        Parameters
        ----------
        values :
            The list of semantic categorical values to calculate its centroid

        **kwargs : optional
            Additional arguments that the specific attribute type value may need to calculate the centroid

        Returns
        -------
        Semantic_categorical_wordnet
            The value that is the differential private centroid of the list of semantic categorical values.
        """
        return Semantic_categorical_wordnet.calculate_centroid(values)

    @staticmethod
    def sort(values):
        """sort

        Sorts the list of semantic categorical values received as parameter.
        In this case, it is not necessary a sorting implementation.

        Parameters
        ----------
        values :
            The list of semantic categorical values to calculate its centroid
        """
        pass

    def calculate_standard_deviation(self, values):
        """calculate_standard_deviation

        Calculates the standard deviation of the list of semantic categorical values received as parameter.

        Parameters
        ----------
        values :
            The list of semantic categorical values to calculate the standard deviation

        Returns
        -------
        Plain_categorical
            The standard deviation of the list of plain categorical values, in this case 0.5.
        """
        return 0.5

    @staticmethod
    def calculate_mean(values):
        """calculate_mean

        Calculates the mean of the list of semantic categorical values received as parameter.
        In this case, the mean consist of the centroid of the values
        Parameters
        ----------
        values :
            The list of semantic categorical values to calculate the mean

        Returns
        -------
        Semantic_categorical_wordnet
            The mean of the list of semantic categorical values.
        """
        mean = Semantic_categorical_wordnet.calculate_centroid(values)

        return mean

    @staticmethod
    def calculate_variance(values):
        """calculate_variance

        Calculates the variance of the list of semantic categorical values received as parameter.
        The variance is calculated in function of the distance to the mean (to the centroid)

        Parameters
        ----------
        values :
            The list of semantic categorical values to calculate the variance

        Returns
        -------
        float
            The variance of the list of semantic categorical values.
        """
        mean = Semantic_categorical_wordnet.calculate_mean(values)
        variance = 0
        for value in values:
            partial = Semantic_categorical_wordnet.similarity_wp_word_syn(
                value.value, mean.syn
            )
            partial = partial * partial
            variance += partial
        variance /= len(values)

        return variance

    @staticmethod
    def calculate_min_max(self, margin):
        """calculate_min_max

        Calculates the min and max value of the list of semantic categorical values received as parameter.
        In this case, there are not max and min values.

        Parameters
        ----------
        values :
            The list of semantic categorical values to calculate the min and max value

        Returns
        -------
        int, int
            The min and max semantic categorical values.
        """
        return None, None

    @staticmethod
    def calculate_reference_value(values):
        """calculate_reference_value

        Calculates the reference value of the list of semantic categorical values received as parameter.
        For semantic categorical, we take the root of the wordnet ontology (entity synset).

        Parameters
        ----------
        values :
            The list of semantic categorical values to calculate the reference value

        Returns
        -------
        Semantic_categorical_wordnet
            The semantic categorical reference value.
        """
        Semantic_categorical_wordnet.reference_value = Semantic_categorical_wordnet(
            "entity"
        )
        Semantic_categorical_wordnet.reference_value.syn = wordnet.synsets("entity")[0]

        return Semantic_categorical_wordnet.reference_value

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.distance(
            Semantic_categorical_wordnet.reference_value
        ) < other.distance(Semantic_categorical_wordnet.reference_value)

    def __str__(self):
        return str(self.value)
