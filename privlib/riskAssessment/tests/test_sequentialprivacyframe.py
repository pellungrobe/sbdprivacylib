from privlib.riskAssessment import constants
from privlib.riskAssessment.sequentialprivacyframe import SequentialPrivacyFrame as SPF
from privlib.riskAssessment.riskevaluators import IndividualSequenceEvaluator, IndividualElementEvaluator
import privlib.riskAssessment.attacks as att
import numpy as np
import pandas as pd
import unittest

class TestSPF(unittest.TestCase):

    def assertDataframeIndexEqual(self, a, b, msg):
        try:
            pd.testing.assert_index_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        self.addTypeEqualityFunc(pd.Index, self.assertDataframeIndexEqual)
        lat_lons = np.array([[43.8430139, 10.5079940],
                             [43.5442700, 10.3261500],
                             [43.7085300, 10.4036000],
                             [43.7792500, 11.2462600],
                             [43.8430139, 10.5079940],
                             [43.7085300, 10.4036000],
                             [43.8430139, 10.5079940],
                             [43.5442700, 10.3261500],
                             [43.5442700, 10.3261500],
                             [43.7085300, 10.4036000],
                             [43.8430139, 10.5079940],
                             [43.7792500, 11.2462600],
                             [43.7085300, 10.4036000],
                             [43.5442700, 10.3261500],
                             [43.7792500, 11.2462600],
                             [43.7085300, 10.4036000],
                             [43.7792500, 11.2462600],
                             [43.8430139, 10.5079940],
                             [43.8430139, 10.5079940],
                             [43.5442700, 10.3261500]])

        traj = pd.DataFrame(lat_lons, columns=['lat', 'lon'])

        traj[constants.DATETIME] = pd.to_datetime([
            '20110203 8:34:04', '20110203 9:34:04', '20110203 10:34:04', '20110204 10:34:04',
            '20110203 8:34:04', '20110203 9:34:04', '20110204 10:34:04', '20110204 11:34:04',
            '20110203 8:34:04', '20110203 9:34:04', '20110204 10:34:04', '20110204 11:34:04',
            '20110204 10:34:04', '20110204 11:34:04', '20110204 12:34:04',
            '20110204 10:34:04', '20110204 11:34:04', '20110205 12:34:04',
            '20110204 10:34:04', '20110204 11:34:04'])

        traj[constants.USER_ID] = [1 for _ in range(4)] + [2 for _ in range(4)] + \
                        [3 for _ in range(4)] + [4 for _ in range(3)] + \
                        [5 for _ in range(3)] + [6 for _ in range(2)]

        lat_lons_2 = np.array([[43.8430139, 10.5079940],
                               [43.8430139, 10.5079940],
                               [43.8430139, 10.5079940],
                               [43.8430139, 10.5079940],
                               [43.8430139, 10.5079940],
                               [43.5442700, 10.3261500],
                               [43.5442700, 10.3261500],
                               [43.5442700, 10.3261500],
                               [43.5442700, 10.3261500],
                               [43.7085300, 10.4036000],
                               [43.7085300, 10.4036000],

                               [43.7792500, 11.2462600],
                               [43.8430139, 10.5079940],
                               [43.7792500, 11.2462600],
                               [43.8430139, 10.5079940],

                               [43.8430139, 10.5079940],
                               [43.8430139, 10.5079940],
                               [43.8430139, 10.5079940],
                               [43.8430139, 10.5079940],
                               [43.8430139, 10.5079940],
                               [43.5442700, 10.3261500],
                               [43.5442700, 10.3261500],
                               [43.5442700, 10.3261500],
                               [43.5442700, 10.3261500],

                               [43.8430139, 10.5079940],
                               [43.8430139, 10.5079940],
                               [43.8430139, 10.5079940],
                               [43.8430139, 10.5079940],
                               [43.8430139, 10.5079940],
                               [43.8430139, 10.5079940],
                               [43.7085300, 10.4036000],
                               [43.5442700, 10.3261500],
                               [43.7085300, 10.4036000],
                               [43.5442700, 10.3261500],
                               [43.7085300, 10.4036000],
                               [43.5442700, 10.3261500]])

        trj2 = pd.DataFrame(lat_lons_2, columns=['lat', 'lng'])

        trj2[constants.DATETIME] = pd.to_datetime(['20110203 8:34:04' for _ in range(36)])

        trj2[constants.USER_ID] = [1 for _ in range(11)] + [2 for _ in range(4)] + \
                        [3 for _ in range(9)] + [4 for _ in range(12)]

        trj2[constants.SEQUENCE_ID] = [1 for _ in range(3)] + [2 for _ in range(8)] + [1 for _ in range(2)] + [2 for _
                                                                                                               in
                                                                                                               range(2)] \
                                      + [1 for _ in range(9)] + [1 for _ in range(4)] + [2 for _ in range(4)] + [3 for _
                                                                                                                 in
                                                                                                                 range(
                                                                                                                     4)]

        self.first_df = traj
        self.second_df = trj2

        self.first_instance = traj[:2].values
        self.second_instance = pd.concat([traj[0:1], traj[3:4]]).values

    def test_spf(self):
        sf = SPF(self.first_df, user_id='uid', datetime='datetime', elements=['lat', 'lon'])
        l1 = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 1, 2]
        self.assertEqual(list(sf.order), l1)
        self.assertEqual(sf.columns, pd.Index(['datetime', 'uid', 'elements', 'sequence', 'order'], dtype='object'))
        l2 = [1,1,1,2,2,2,2,2,2,2,2,1,1,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,3]
        sec_sf = SPF(self.second_df, user_id='uid', datetime='datetime', elements=['lat', 'lng'], sequence_id='seq')
        self.assertEqual(list(sec_sf.sequence),l2)

    def test_casescomputation(self):
        sf = SPF(self.second_df, user_id='uid', datetime='datetime', elements=['lat', 'lng'], sequence_id='seq')
        iee = IndividualElementEvaluator(sf, att.SequenceAttack, 3)
        a = iee.assess_risk(complete=True)
        self.assertEqual(len(a), 473)
        self.assertEqual(list(a.case_risk.unique()), [0.33333333333333333333, 0.5, 1])

    def test_elementattack(self):
        sf = SPF(self.second_df, user_id='uid', datetime='datetime', elements=['lat', 'lng'], sequence_id='seq')
        iee = IndividualElementEvaluator(sf, att.ElementsAttack, 2)
        ise = IndividualSequenceEvaluator(sf, att.ElementsAttack, 2)
        a = iee.assess_risk()
        b = ise.assess_risk(complete=True)
        ra = [0.5, 1.0, 0.3333333333333333, 0.5]
        rb = [0.5 for _ in range(31)]
        rb.extend([1.0 for _ in range(2)])
        rb.extend([0.3333333333333333 for _ in range(36)])
        rb.extend([0.6666666666666666 for _ in range(18)])
        self.assertEqual(a['risk'].to_list(), ra)
        self.assertEqual(b['risk'].to_list(),rb)

    def test_sequenceattack(self):
        sf = SPF(self.second_df, user_id='uid', datetime='datetime', elements=['lat', 'lng'], sequence_id='seq')
        iee = IndividualElementEvaluator(sf, att.SequenceAttack, 3)
        ise = IndividualSequenceEvaluator(sf, att.SequenceAttack, 3)
        a = iee.assess_risk()
        b = ise.assess_risk()
        ra = [1.0, 1.0, 0.3333333333333333, 1.0]
        rb = [1.0, 1.0, 0.5, 1.0]
        self.assertEqual(a['risk'].to_list(), ra)
        self.assertEqual(b['risk'].to_list(), rb)

    if __name__ == '__main__':
        unittest.main()