from privlib.discriminationDiscovery.discrimination_discovery import *
import numpy as np
import pandas as pd
import unittest
class TestDD(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv("data/credit.csv", sep=',', na_values='?')

    def test_dd(self):
        disc = DD(self.df, 'foreign_worker=no', 'class=bad')
        ctgs = disc.extract(testCond=check_rd, minSupp=-20, maxn=100)
        r = [0.42572815533980585,0.4145669291338583]
        for i in range(len(ctgs[:2])):
            v, ctg = ctgs[i]
            self.assertEqual(v,r[i])

    if __name__ == '__main__':
        unittest.main()