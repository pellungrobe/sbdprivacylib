from antiDiscrimination.src.algorithms.anonymization_scheme import Anonymization_scheme
# from apyori import apriori
import itertools
from tqdm.auto import tqdm
import pickle
import copy
from antiDiscrimination.src.entities.anti_discrimination_metrics import Anti_discrimination_metrics


class Anti_discrimination(Anonymization_scheme):
    """Anti_discrimination

    Class that implements anti-discrimination anonymization.
    This algorithm implementation can be executed by the anonymization scheme due to its extends
    Anonymization_scheme class
    (See examples of use in sections 1 and 2 of the jupyter notebook: test_antiDiscrimination.ipynb)
    (See also the file "anti_discrimination_test.py" in the folder "tests")

    See Also
    --------
    :class:`Anonymization_scheme`

    References
    ----------
    .. [1] Sara Hajian and Josep Domingo-Ferrer, "A methodology for direct and indirect discrimination prevention in
           data mining", IEEE Transactions on Knowledge and Data Engineering, Vol. 25, no. 7, pp. 1445-1459, Jun 2013.
           DOI: https://doi.org/10.1109/TKDE.2012.72
    """
    hash_num_rec = {}

    def __init__(self, original_dataset, min_support, min_confidence, alfa, DI):
        """Constructor, called from inherited classes

        Parameters
        ----------
        original_dataset : :class:`Dataset`
            the data set to be anonymized.

        min_support : float
            min support to consider a rule a frequent rule

        min_confidence : float
            min confidence to consider a rule a frequent rule

        alfa : float
            discriminatory threshold to consider a frequent rule a direct or an indirect discrimination rule

        DI : list of tuples
            Predetermined discriminatory items, each tuple corresponds to: (attribute,discriminatory value)

        See Also
        --------
        :class:`Dataset`
        """
        super().__init__(original_dataset)
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.alfa = alfa
        self.DI = self.to_item_DI(DI)
        # todo: now, last attribute is the class, change to read the class from settings (it has to be the last one)
        self.index_class = self.original_dataset.num_attr - 1
        self.FR_rules = []
        self.MR_rules = []
        self.PR_rules = []
        self.RR_rules = []
        self.non_RR_rules = []

    def calculate_anonymization(self):
        """calculate_anonymization

        Function to perform the anti-discrimination anonymization.
        """
        print("Anonymizing " + str(self))
        print("Alfa = " + str(self.alfa))
        self.FR_rules, PD_rules, PND_rules = self.create_rules(self.original_dataset, self.index_class)
        self.RR_rules, self.non_RR_rules = self.calculate_RR_rules(self.original_dataset, PND_rules)
        self.MR_rules, self.PR_rules = self.calculate_MR_rules(self.original_dataset, PD_rules)
        self.anonymized_dataset = copy.deepcopy(self.original_dataset)
        self.anonymize_direct_indirect()

    # def calculate_anonymization(self, algorithm=None):
    #     print("Anonymizing " + str(self))
    #     # self.calculate_and_save_rules_direct()
    #     self.FR_rules,_,_ = self.load_rules_FR()
    #     self.MR_rules, self.PR_rules = self.load_rules_direct()
    #     self.anonymized_dataset = copy.deepcopy(self.original_dataset)
    #     self.anonymize_direct_rules()

    # def calculate_anonymization(self):
    #     print("Anonymizing " + str(self))
    #     # self.calculate_and_save_rules_indirect()
    #     self.FR_rules,_,_ = self.load_rules_FR()
    #     self.RR_rules, self.non_RR_rules = self.load_rules_indirect()
    #     self.anonymized_dataset = copy.deepcopy(self.original_dataset)
    #     self.anonymize_indirect_rules()

    def calculate_and_save_rules_direct(self):
        self.calculate_and_save_FR_rules()
        self.FR_rules, PD_rules, PND_rules = self.load_rules_FR()
        self.MR_rules, self.PR_rules = self.calculate_MR_rules(self.original_dataset, PD_rules)
        self.save_rules_direct(self.MR_rules, self.PR_rules)

    def calculate_and_save_rules_indirect(self):
        self.calculate_and_save_FR_rules()
        self.FR_rules, PD_rules, PND_rules = self.load_rules_FR()
        self.RR_rules, self.non_RR_rules = self.calculate_RR_rules(self.original_dataset, PND_rules)
        self.save_rules_indirect(self.RR_rules, self.non_RR_rules)

    def calculate_and_save_FR_rules(self):
        self.FR_rules, PD_rules, PND_rules = self.create_rules(self.original_dataset, self.index_class)
        self.save_rules_FR(self.FR_rules, PD_rules, PND_rules)

    def anonymize_indirect_rules(self):
        records = []
        for record in self.anonymized_dataset.records:
            records.append([str(record.values[i]) for i in range(0, self.anonymized_dataset.num_attr)])
        total_records = len(records)
        print("Anonymizing...")
        for RR_rule in tqdm(self.RR_rules):
            gamma = RR_rule[0].confidence
            X = RR_rule[0].premise
            C = RR_rule[0].consequence
            for ABC_rule in RR_rule[1]:
                num_rule = self.count_items_no_hash(records, ABC_rule.rule)
                support_ABC = num_rule / total_records
                AB = ABC_rule.A + ABC_rule.B
                num_rule = self.count_items_no_hash(records, AB)
                support_AB = num_rule / total_records
                confidence_ABC = support_ABC / support_AB
                A = ABC_rule.A
                XA = X + A
                num_rule = self.count_items_no_hash(records, XA)
                support_XA = num_rule / total_records
                num_rule = self.count_items_no_hash(records, X)
                support_X = num_rule / total_records
                confidence_XA = support_XA / support_X
                beta2 = confidence_XA
                delta1 = support_XA
                B = ABC_rule.B
                BC = B + C
                num_rule = self.count_items_no_hash(records, BC)
                support_BC = num_rule / total_records
                num_rule = self.count_items_no_hash(records, B)
                support_B = num_rule / total_records
                confidence_BC = support_BC / support_B
                delta = confidence_BC
                BA = B + A
                num_rule = self.count_items_no_hash(records, BA)
                support_BA = num_rule / total_records
                delta2 = support_BA
                beta1 = delta1 / delta2
                D = ABC_rule.D
                DBc = self.get_noA_B_noD_noC(records, A, B, D, C)
                DBc_impact = []
                for dbc in DBc:
                    record = records[dbc]
                    impact = self.calculate_impact(self.FR_rules, record)
                    DBc_impact.append([dbc, impact])
                DBc_impact.sort(key=lambda x: x[1])
                while delta <= (beta1 * (beta2 + gamma - 1)) / (beta2 * self.alfa):
                    if len(DBc_impact) > 0:
                        first_dbc = DBc_impact.pop(0)
                        records[first_dbc[0]][self.index_class] = C[0].item
                        self.anonymized_dataset.records[first_dbc[0]].values[self.index_class] = C[0].item
                        num_rule = self.count_items_no_hash(records, BC)
                        support_BC = num_rule / total_records
                        confidence_BC = support_BC / support_B
                        delta = confidence_BC
                    else:
                        break

    def anonymize_direct_rules(self):
        records = []
        for record in self.anonymized_dataset.records:
            records.append([str(record.values[i]) for i in range(0, self.anonymized_dataset.num_attr)])
        total_records = len(records)
        print("Anonymizing...")
        for ABC_rule in tqdm(self.MR_rules):
            self.FR_rules.remove(ABC_rule)
            A = ABC_rule.A
            B = ABC_rule.B
            C = ABC_rule.consequence
            DBc = self.get_noA_B_noC(records, A, B, C)
            DBc_impact = []
            for dbc in DBc:
                record = records[dbc]
                impact = self.calculate_impact(self.FR_rules, record)
                DBc_impact.append([dbc, impact])
            DBc_impact.sort(key=lambda x: x[1])
            BC = B + C
            num_rule = self.count_items_no_hash(records, BC)
            support_BC = num_rule / total_records
            num_rule = self.count_items_hash(records, B)
            support_B = num_rule / total_records
            confidence_BC = support_BC / support_B
            delta = confidence_BC
            confidence_ABC = ABC_rule.confidence
            cond = confidence_ABC / self.alfa
            while delta <= cond:
                first_dbc = DBc_impact.pop(0)
                records[first_dbc[0]][self.index_class] = C[0].item
                self.anonymized_dataset.records[first_dbc[0]].values[self.index_class] = C[0].item
                num_rule = self.count_items_no_hash(records, BC)
                support_BC = num_rule / total_records
                confidence_BC = support_BC / support_B
                delta = confidence_BC

    def anonymize_direct_indirect(self):
        print("Anonymizing " + str(self))
        records = []
        for record in self.anonymized_dataset.records:
            records.append([str(record.values[i]) for i in range(0, self.anonymized_dataset.num_attr)])
        total_records = len(records)
        record_impact = []
        print("Calculating impacts...")
        for record in tqdm(records):
            impact = self.calculate_impact(self.FR_rules, record)
            record_impact.append(impact)
        print("Anonymizing...")
        for RR_rule in tqdm(self.RR_rules):
            gamma = RR_rule[0].confidence
            X = RR_rule[0].premise
            C = RR_rule[0].consequence
            for ABC_rule in RR_rule[1]:
                num_rule = self.count_items_no_hash(records, ABC_rule.rule)
                support_ABC = num_rule / total_records
                AB = ABC_rule.A + ABC_rule.B
                num_rule = self.count_items_no_hash(records, AB)
                support_AB = num_rule / total_records
                confidence_ABC = support_ABC / support_AB
                A = ABC_rule.A
                XA = X + A
                num_rule = self.count_items_no_hash(records, XA)
                support_XA = num_rule / total_records
                num_rule = self.count_items_no_hash(records, X)
                support_X = num_rule / total_records
                confidence_XA = support_XA / support_X
                beta2 = confidence_XA
                delta1 = support_XA
                B = ABC_rule.B
                BC = B + C
                num_rule = self.count_items_no_hash(records, BC)
                support_BC = num_rule / total_records
                num_rule = self.count_items_no_hash(records, B)
                support_B = num_rule / total_records
                confidence_BC = support_BC / support_B
                delta = confidence_BC
                BA = B + A
                num_rule = self.count_items_no_hash(records, BA)
                support_BA = num_rule / total_records
                delta2 = support_BA
                beta1 = delta1 / delta2
                D = ABC_rule.D
                DBc = self.get_noA_B_noD_noC(records, A, B, D, C)
                DBc_impact = []
                for dbc in DBc:
                    impact = record_impact[dbc]
                    DBc_impact.append([dbc, impact])
                DBc_impact.sort(key=lambda x: x[1])
                if ABC_rule in self.MR_rules:
                    # if self.is_rule_in_rule_set(ABC_rule, MR_rules):
                    while delta <= (beta1 * (beta2 + gamma - 1)) / (beta2 * self.alfa) and \
                            delta <= (confidence_ABC / self.alfa):
                        if len(DBc_impact) > 0:
                            first_dbc = DBc_impact.pop(0)
                            records[first_dbc[0]][self.index_class] = C[0].item
                            self.anonymized_dataset.records[first_dbc[0]].values[self.index_class] = C[0].item
                            num_rule = self.count_items_no_hash(records, BC)
                            support_BC = num_rule / total_records
                            confidence_BC = support_BC / support_B
                            delta = confidence_BC
                        else:
                            break
                else:
                    while delta <= (beta1 * (beta2 + gamma - 1)) / (beta2 * self.alfa):
                        if len(DBc_impact) > 0:
                            first_dbc = DBc_impact.pop(0)
                            records[first_dbc[0]][self.index_class] = C[0].item
                            self.anonymized_dataset.records[first_dbc[0]].values[self.index_class] = C[0].item
                            num_rule = self.count_items_no_hash(records, BC)
                            support_BC = num_rule / total_records
                            confidence_BC = support_BC / support_B
                            delta = confidence_BC
                        else:
                            break
        RR_rules_temp = []
        for RR_rule in self.RR_rules:
            RR_rules_temp.append(RR_rule[0])
            for ABC_rule in RR_rule[1]:
                RR_rules_temp.append(ABC_rule)
        MR_noRR_rules = []
        for MR_rule in self.MR_rules:
            # if MR_rule not in RR_rules_temp:
            #     MR_noRR_rules.append(MR_rule)
            MR_noRR_rules.append(MR_rule)
        for ABC_rule in tqdm(MR_noRR_rules):
            A = ABC_rule.A
            B = ABC_rule.B
            C = ABC_rule.consequence
            num_rule = self.count_items_no_hash(records, ABC_rule.rule)
            support_ABC = num_rule / total_records
            AB = ABC_rule.A + ABC_rule.B
            num_rule = self.count_items_hash(records, AB)
            support_AB = num_rule / total_records
            confidence_ABC = support_ABC / support_AB
            BC = B + C
            num_rule = self.count_items_no_hash(records, BC)
            support_BC = num_rule / total_records
            num_rule = self.count_items_hash(records, B)
            support_B = num_rule / total_records
            confidence_BC = support_BC / support_B
            delta = confidence_BC
            DBc = self.get_noA_B_noC(records, A, B, C)
            DBc_impact = []
            for dbc in DBc:
                impact = record_impact[dbc]
                DBc_impact.append([dbc, impact])
            DBc_impact.sort(key=lambda x: x[1])
            while delta <= (confidence_ABC / self.alfa):
                # print(f"delta: {delta} <= {confidence_ABC/self.alfa} quedan: {len(DBc_impact)}")
                if len(DBc_impact) > 0:
                    first_dbc = DBc_impact.pop(0)
                    records[first_dbc[0]][self.index_class] = C[0].item
                    self.anonymized_dataset.records[first_dbc[0]].values[self.index_class] = C[0].item
                    num_rule = self.count_items_no_hash(records, BC)
                    support_BC = num_rule / total_records
                    confidence_BC = support_BC / support_B
                    delta = confidence_BC
                else:
                    break

    @staticmethod
    def is_rule_in_rule_set(rule, rule_set):
        for r in rule_set:
            if len(r.rule) != len(rule.rule):
                continue
            for item in r:
                ok = True
                if not item in rule.rule:
                    ok = False
                    break
            if ok:
                return True

        return False

    def get_noA_B_noD_noC(self, records, A, B, D, C):
        noA_B_noD_noC_records = []
        for ind, record in enumerate(records):
            if self.is_any_item_set_in_record(record, A):
                continue
            if not self.is_all_item_set_in_record(record, B):
                continue
            if self.is_any_item_set_in_record(record, D):
                continue
            if self.is_any_item_set_in_record(record, C):
                continue
            noA_B_noD_noC_records.append(ind)

        return noA_B_noD_noC_records

    def get_noA_B_noC(self, records, A, B, C):
        noA_B_noC_records = []
        for ind, record in enumerate(records):
            if self.is_any_item_set_in_record(record, A):
                continue
            if not self.is_all_item_set_in_record(record, B):
                continue
            if self.is_any_item_set_in_record(record, C):
                continue
            noA_B_noC_records.append(ind)

        return noA_B_noC_records

    def calculate_metrics(self):
        print("Calculating metrics on anonymized dataset...")
        # todo: now, last attribute is the class, change to read the class from settings (it has to be the last one)
        index_class = self.anonymized_dataset.num_attr - 1
        FR_rules_a, PD_rules_a, PND_rules_a = self.create_rules(self.anonymized_dataset, index_class)
        RR_rules_a, non_RR_rules_a = self.calculate_RR_rules(self.anonymized_dataset, PND_rules_a)
        RR_rules_a = [rule for rule in RR_rules_a if rule in self.RR_rules]
        MR_rules_a, PR_rules_a = self.calculate_MR_rules(self.anonymized_dataset, PD_rules_a)
        MR_rules_a = [rule for rule in MR_rules_a if rule in self.MR_rules]

        if len(self.RR_rules) > 0:
            IDPD = (len(self.RR_rules) - len(RR_rules_a)) / len(self.RR_rules)
        else:
            IDPD = 1.0
        intersection = [rule for rule in self.non_RR_rules if rule in non_RR_rules_a]
        IDPP = len(intersection) / len(self.non_RR_rules)
        if len(self.MR_rules) > 0:
            DDPD = (len(self.MR_rules) - len(MR_rules_a)) / len(self.MR_rules)
        else:
            DDPD = 1.0
        intersection = [rule for rule in self.PR_rules if rule in PR_rules_a]
        DDPP = len(intersection) / len(self.PR_rules)

        anti_discrimination_metrics = Anti_discrimination_metrics(DDPD, DDPP, IDPD, IDPP)

        return anti_discrimination_metrics

    def calculate_MR_rules(self, dataset, PD_rules):
        print("Calculating MR and PR rules...")
        records = []
        for record in dataset.records:
            records.append([str(record.values[i]) for i in range(0, self.original_dataset.num_attr)])
        total_records = len(records)
        MR_rules = []
        PR_rules = []
        for rule in tqdm(PD_rules):
            A = []
            B = []
            for item in rule.premise:
                if item in self.DI:
                    A.append(item)
                else:
                    B.append(item)
            rule.A = A[:]
            rule.B = B[:]
            confidence_ABC = rule.confidence
            BC = B + rule.consequence
            num_rule = self.count_items_hash(records, BC)
            support_BC = num_rule / total_records
            num_rule = self.count_items_hash(records, B)
            support_B = num_rule / total_records
            confidence_BC = support_BC / support_B
            elif_ABC = confidence_ABC / confidence_BC
            if elif_ABC >= self.alfa:
                MR_rules.append(rule)
            else:
                PR_rules.append(rule)
        print("MR Rules: " + str(len(MR_rules)))
        print("PR Rules: " + str(len(PR_rules)))

        return MR_rules, PR_rules

    def calculate_RR_rules(self, dataset, PND_rules):
        print("Calculating RR and non_RR rules...")
        records = []
        for record in dataset.records:
            records.append([str(record.values[i]) for i in range(0, self.original_dataset.num_attr)])
        total_records = len(records)
        RR_rules = []
        non_RR_rules = []
        for PND_rule in tqdm(PND_rules):
            ABC_rules = []
            is_PND_rule_RR = False
            for num_items_premise in range(1, len(PND_rule.premise) + 1):
                for permutation_premise in itertools.permutations(PND_rule.premise, num_items_premise):
                    D = []
                    B = PND_rule.premise[:]
                    C = PND_rule.consequence
                    for item in permutation_premise:
                        D.append(item)
                        B.remove(item)
                    confidence_DBC = PND_rule.confidence
                    BC = B + C
                    num_rule = self.count_items_hash(records, BC)
                    support_BC = num_rule / total_records
                    num_rule = self.count_items_hash(records, B)
                    support_B = num_rule / total_records
                    confidence_BC = support_BC / support_B
                    # Search all A combinations with the premise DB to form DBA
                    for num_items_DI in range(1, len(self.DI) + 1):
                        for comb_DI in itertools.combinations(self.DI, num_items_DI):
                            A = []
                            for item in comb_DI:
                                A.append(item)
                            DBA = PND_rule.premise + A
                            num_rule = self.count_items_hash(records, DBA)
                            support_DBA = num_rule / total_records
                            DB = PND_rule.premise
                            num_rule = self.count_items_hash(records, DB)
                            support_DB = num_rule / total_records
                            confidence_DBA = support_DBA / support_DB
                            BA = B + A
                            num_rule = self.count_items_hash(records, BA)
                            support_BA = num_rule / total_records
                            if support_BA == 0:
                                continue
                            confidence_ABD = support_DBA / support_BA
                            gamma = confidence_DBC
                            delta = confidence_BC
                            beta1 = confidence_ABD
                            beta2 = confidence_DBA
                            if beta2 == 0:
                                continue
                            elb = self.elb(gamma, delta, beta1, beta2)
                            if elb >= self.alfa:
                            # if elb >= 1.0:
                                is_PND_rule_RR = True
                                ABC_rule = Rule(A + B, C, None, None)
                                ABC_rule.A = A[:]
                                ABC_rule.B = B[:]
                                ABC_rule.D = D[:]
                                ABC_rules.append(ABC_rule)
            if is_PND_rule_RR:
                RR = [PND_rule, ABC_rules]
                RR_rules.append(copy.deepcopy(RR))
                # print(len(RR_rules))
                # print(RR_rules)
            else:
                non_RR_rules.append(PND_rule)

        count_IR = 0
        for RR_rule in RR_rules:
            count_IR += len(RR_rule[1])
        print("RR Rules: " + str(len(RR_rules)))
        print("Indirect alfa-discriminatory rules: " + str(count_IR))
        print("non RR Rules: " + str(len(non_RR_rules)))

        return RR_rules, non_RR_rules

    def elb(self, x, y, b1, b2):
        f = self.f(x, b1, b2)
        if f <= 0:
            return 0
        return f / y

    def f(self, x, b1, b2):
        return (b1 / b2) * (b2 + x - 1)

    def create_rules(self, dataset, index_class):
        print("Calculating FR rules...")
        Anti_discrimination.hash_num_rec = {}
        len_item_set = dataset.num_attr
        records = []
        for record in dataset.records:
            records.append([str(record.values[i]) for i in range(0, len_item_set)])
        total_records = len(records)
        items_temp = []
        for i in range(0, len_item_set - 1):
            set_values = set()
            for record in records:
                item = Item(record[i], i)
                set_values.add(item)
            items_temp.append(list(set_values))

        items = []
        for i in range(0, len_item_set - 1):
            values = []
            for item in items_temp[i]:
                list_items = [item]
                count = self.count_items_hash(records, list_items)
                support = count / total_records
                if support >= self.min_support:
                    values.append(item)
            items.append(values)

        clas = set()
        for record in records:
            item = Item(record[index_class], index_class)
            clas.add(item)
        clas = list(clas)

        items_DI = []
        for item in self.DI:
            items_DI.append(item)

        list_attr = [i for i in range(len(items))]
        total_iter = 0
        for num_attr in range(1, len(list_attr) + 1):
            for comb in itertools.combinations(list_attr, num_attr):
                num_iter = 1
                for i in comb:
                    num_iter *= len(items[i])
                total_iter += num_iter
        pbar = tqdm(total=total_iter)
        control = {"[]": True}
        count = 0
        count_BK = 0
        FR_rules = []
        PD_rules = []
        PND_rules = []
        BK_rules = []
        for num_attr in range(1, len(list_attr) + 1):
            for comb in itertools.combinations(list_attr, num_attr):
                items_comb = []
                for i in comb:
                    items_comb.append(items[i])
                maxs = self.calculate_maxs_index(items_comb)
                index = self.calculate_next_index(None, maxs)
                while index is not None:
                    X = []
                    for i in range(len(items_comb)):
                        X.append(items_comb[i][index[i]])
                    num_X = self.is_rule_possible(records, X, control, self.min_support)
                    if num_X is None:
                        pbar.update(1)
                        index = self.calculate_next_index(index, maxs)
                        continue
                    for c in clas:
                        rule = X[:]
                        rule.append(c)
                        num_rule = self.count_items_hash(records, rule)
                        support = num_rule / total_records
                        if support >= self.min_support:
                            confidence = num_rule / num_X
                            if confidence >= self.min_confidence:
                                rule = Rule(X, [c], support, confidence)
                                FR_rules.append(rule)
                                if self.is_PD_rule(X):
                                    PD_rules.append(rule)
                                    # print("PD_rule: " + str(rule) + " (" + str(count) + ")")
                                else:
                                    PND_rules.append(rule)
                                    # print("PND_rule: " + str(rule) + " (" + str(count) + ")")
                                count += 1
                    pbar.update(1)
                    index = self.calculate_next_index(index, maxs)

        pbar.close()
        print("FR Rules: " + str(len(FR_rules)))
        print("PD Rules: " + str(len(PD_rules)))
        print("PND Rules: " + str(len(PND_rules)))
        print("Total FR = PD + PND: " + str(len(PD_rules) + len(PND_rules)))

        return FR_rules, PD_rules, PND_rules

    def is_PD_rule(self, X):
        for item in X:
            if item in self.DI:
                return True
        return False

    @staticmethod
    def is_any_A_in_X(X, A):
        for item in X:
            if item in A:
                return True
        return False

    # def create_rules_FR_apriori(self):
    #     t_ini = timer()
    #     print("Anonymizing " + str(self))
    #     # todo: now, last attribute is the class, change to read the class from settings
    #     index_class = self.original_dataset.num_attr - 1
    #     set_class = set()
    #     for record in self.original_dataset.records:
    #         set_class.add(str(record.values[index_class]))
    #     len_item_set = self.original_dataset.num_attr
    #     records = []
    #     for record in self.original_dataset.records:
    #         records.append([str(record.values[i]) for i in range(0, len_item_set)])
    #     association_rules = apriori(records, min_support=0.02, min_confidence=0.1)
    #     cont = 0
    #     for rule in association_rules:
    #         X = list(rule[2][0][0])
    #         C = list(rule[2][0][1])
    #         if len(C) != 1 or C[0] not in set_class:
    #             # if C[0] not in set_class:
    #             continue
    #         # if len(X) < 2:
    #         #     continue
    #         cont += 1
    #         support = rule[1]
    #         confidence = rule[2][0][2]
    #         lift = rule[2][0][3]
    #         print(f"{X} -> {C} (supp: {support}, conf: {confidence}, lift: {lift}, cont: {cont} )")
    #         # print(rule)
    #     print(cont)
    #     association_results = list(association_rules)
    #     print(len(association_results))
    #     print(association_rules[0])
    #     rules = Anti_discrimination.inspect(association_results)
    #     print(rules)

    def to_item_DI(self, DI):
        temp = []
        for di in DI:
            index = self.original_dataset.header.index(di[0])
            item = Item(di[1], index)
            temp.append(item)

        return temp

    @staticmethod
    def save_rules_direct(MR_rules, PR_rules):
        with open("../../input_datasets/rules_MR_PR.dat", 'wb') as fp:
            pickle.dump(MR_rules, fp)
            pickle.dump(PR_rules, fp)
            pickle.dump(Anti_discrimination.hash_num_rec, fp)

    @staticmethod
    def save_rules_indirect(RR_rules, non_RR_rules):
        with open("../../input_datasets/rules_RR_nonRR.dat", 'wb') as fp:
            pickle.dump(RR_rules, fp)
            pickle.dump(non_RR_rules, fp)
            pickle.dump(Anti_discrimination.hash_num_rec, fp)

    @staticmethod
    def load_rules_direct():
        with open("../../input_datasets/rules_MR_PR.dat", 'rb') as fp:
            MR_rules = pickle.load(fp)
            PR_rules = pickle.load(fp)
            Anti_discrimination.hash_num_rec = pickle.load(fp)
        print("MR Rules loaded: " + str(len(MR_rules)))
        print("PR Rules loaded: " + str(len(PR_rules)))

        return MR_rules, PR_rules

    @staticmethod
    def load_rules_indirect():
        with open("../../input_datasets/rules_RR_nonRR.dat", 'rb') as fp:
            RR_rules = pickle.load(fp)
            non_RR_rules = pickle.load(fp)
            Anti_discrimination.hash_num_rec = pickle.load(fp)
        print("RR Rules loaded: " + str(len(RR_rules)))
        print("non RR Rules loaded: " + str(len(non_RR_rules)))

        return RR_rules, non_RR_rules

    @staticmethod
    def save_rules_FR(FR_rules, PD_rules, PND_rules):
        with open("../../input_datasets/rules_FR_PD_PND.dat", 'wb') as fp:
            pickle.dump(FR_rules, fp)
            pickle.dump(PD_rules, fp)
            pickle.dump(PND_rules, fp)
            pickle.dump(Anti_discrimination.hash_num_rec, fp)

    @staticmethod
    def load_rules_FR():
        with open("../../input_datasets/rules_FR_PD_PND.dat", 'rb') as fp:
            FR_rules = pickle.load(fp)
            PD_rules = pickle.load(fp)
            PND_rules = pickle.load(fp)
            Anti_discrimination.hash_num_rec = pickle.load(fp)
        print("FR Rules loaded: " + str(len(FR_rules)))
        print("PD Rules loaded: " + str(len(PD_rules)))
        print("PND Rules loaded: " + str(len(PND_rules)))
        print("Total FR = PD + PND: " + str(len(PD_rules) + len(PND_rules)))

        return FR_rules, PD_rules, PND_rules

    @staticmethod
    def is_rule_possible(records, X, control, min_support):
        p = str(X[0:-1])
        if p not in control:
            return None
        num_X = Anti_discrimination.count_items_hash(records, X)
        support = num_X / len(records)
        if support < min_support:
            return None
        p = str(X)
        control[p] = True

        return num_X

    @staticmethod
    def count_items_hash(records, items):
        key = str(items)
        if key in Anti_discrimination.hash_num_rec:
            return Anti_discrimination.hash_num_rec[key]
        count = 0
        for record in records:
            ok = True
            for item in items:
                if record[item.index].lower() != item.item.lower():
                    ok = False
                    break
            if ok:
                count += 1
        Anti_discrimination.hash_num_rec[key] = count

        return count

    @staticmethod
    def count_items_no_hash(records, items):
        count = 0
        for record in records:
            ok = True
            for item in items:
                if record[item.index].lower() != item.item.lower():
                    ok = False
                    break
            if ok:
                count += 1

        return count

    @staticmethod
    def calculate_impact(FR_rules, record):
        count = 0
        for rule in FR_rules:
            ok = True
            for item in rule.premise:
                if record[item.index].lower() != item.item.lower():
                    ok = False
                    break
            if ok:
                count += 1

        return count

    @staticmethod
    def is_all_item_set_in_record(record, item_set):
        ok = True
        for item in item_set:
            if record[item.index].lower() != item.item.lower():
                ok = False
                break
        if ok:
            return True

        return False

    @staticmethod
    def is_any_item_set_in_record(record, item_set):
        for item in item_set:
            if record[item.index].lower() == item.item.lower():
                return True

        return False

    @staticmethod
    def calculate_next_index(index, maxs):
        if index is None:
            index = [0 for i in maxs]
            return index
        index_final = len(index) - 1
        index[index_final] += 1
        if index[index_final] == maxs[index_final]:
            index[index_final] = 0
            ok = False
            for i in range(len(index) - 2, -1, -1):
                index[i] += 1
                if index[i] < maxs[i]:
                    ok = True
                    break
                else:
                    index[i] = 0
            if ok is False:
                return None

        return index

    @staticmethod
    def calculate_maxs_index(items):
        maxs = []
        for i in range(len(items)):
            maxs.append(len(items[i]))

        return maxs

    @staticmethod
    def inspect(results):
        rh = [tuple(result[2][0][0]) for result in results]
        lh = [tuple(result[2][0][1]) for result in results]
        supports = [result[1] for result in results]
        confidences = [result[2][0][2] for result in results]
        lifts = [result[2][0][3] for result in results]

        return list(zip(rh, lh, supports, confidences, lifts))

    def __str__(self):
        return "anti-discrimination via direct & indirect discrimination detection"


class Rule:
    def __init__(self, premise, consequence, support, confidence):
        self.premise = premise
        self.consequence = consequence
        self.rule = self.premise + self.consequence
        self.A = []
        self.B = []
        self.D = []
        self.support = support
        self.confidence = confidence

    # they have to be ordered by attribute order in record
    # def __eq__(self, rule):
    #     if len(self.rule) != len(rule.rule):
    #         return False
    #     for ind, item in enumerate(self.rule):
    #         if item != rule.rule[ind]:
    #             return False
    #     return True

    # they can be unordered
    def __eq__(self, rule):
        if len(self.rule) != len(rule.rule):
            return False
        for item in self.rule:
            if not item in rule.rule:
                return False

        return True

    def __str__(self):
        s = str(self.premise)
        s += " -> " + str(self.consequence)
        s += " (support: " + str(self.support)
        s += " confidence: " + str(self.confidence)
        s += ")"

        return s

    def __repr__(self):
        return str(self)


class Item:
    def __init__(self, item, index):
        self.item = item
        self.index = index

    def __eq__(self, item):
        return self.item.lower() == item.item.lower() and self.index == item.index

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return self.item + "(" + str(self.index) + ")"

    def __repr__(self):
        return str(self)
