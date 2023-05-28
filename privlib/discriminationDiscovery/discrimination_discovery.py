# -*- coding: utf-8 -*-
"""
dd Discrimination Discocery
version 1.0

@author: Salvatore Ruggieri
"""

import numpy as np
import pandas as pd
import pyroaring
import csv
import fim
import sys
import urllib
import gzip
import codecs
import queue as Q

"""
 Return a reader from a file, url, or gzipped file/url
"""


def getReader(filename, encoding="utf8"):
    if filename == "":
        return sys.stdin
    try:
        if filename.endswith(".gz"):
            file = gzip.open(filename)
        else:
            file = open(filename, encoding=encoding)
    except:
        file = urllib.request.urlopen(filename)
        if filename.endswith(".gz"):
            file = gzip.GzipFile(fileobj=file)
        reader = codecs.getreader(encoding)
        file = reader(file)
    return file


"""
 Return the list of attributes in the header of a CSV or ARFF input reader.
"""


def getCSVattributes(input, sep=","):
    result = []
    line = input.readline()
    if line.startswith("@relation"):
        for line in input:
            if line.startswith("@data"):
                break
            else:
                if line.startswith("@attribute"):
                    result.append(line.split(" ")[1])
    else:
        result = line.strip().split(sep)
    return result


""" Extract attribute name from attribute=value string. """


def get_att(itemDesc):
    pos = itemDesc.find("=")
    if pos >= 0:
        return itemDesc[:pos]
    else:
        return ""


"""
 Read a CSV/ARFF file and code attribute=value as an item.
 Returns a list of transactions, plus coding and decoding dictionaries
"""


def CSV2tranDB(filename, sep=",", na_values="?", one_hot_set=None):
    with getReader(filename) as inputf:
        # header
        attributes = getCSVattributes(inputf, sep=sep)
        if one_hot_set is None:
            one_hot_set = set(a for a in attributes if get_att(a) != "")
        # reader for the rest of the file
        csvreader = csv.reader(inputf, delimiter=sep)
        nitems = 0
        codes = {}
        tDB = []
        # scan rows in CSV
        for values in csvreader:
            if len(values) == 0:
                continue
            # create transaction
            transaction = []
            for att, item in zip(attributes, values):
                if item == na_values:
                    continue
                if item == 0 and att in one_hot_set:
                    continue
                attitem = att + "=" + item
                code = codes.get(attitem)  # code of attitem
                if code is None:
                    codes[attitem] = code = nitems
                    nitems += 1
                transaction.append(code)
            # append transaction
            tDB.append(transaction)
    # decode list
    decodes = {code: attitem for attitem, code in codes.items()}
    return (tDB, codes, decodes)


"""
 Read a CSV/ARFF file and code attribute=value as an item.
 Returns a list of transactions, plus coding and decoding dictionaries
"""


def PD2tranDB(df, na_values="NaN", one_hot_set=None):
    nitems = 0
    codes = {}
    tDB = []
    if one_hot_set is None:
        one_hot_set = set(a for a in df.columns.to_list() if get_att(a) != "")
    # print(one_hot_set)
    for _, row in df.iterrows():
        transaction = []
        for att, item in row.items():
            if item == na_values:
                continue
            if item == 0 and att in one_hot_set:
                continue
            attitem = att + "=" + str(item)
            code = codes.get(attitem)  # code of attitem
            if code is None:
                codes[attitem] = code = nitems
                nitems += 1
            transaction.append(code)
        # append transaction
        tDB.append(np.array(transaction, dtype=int))
    # decode list
    # for k,v in codes.items():
    #    print(k, v)
    # print(codes.items())
    decodes = {code: attitem for attitem, code in codes.items()}
    return (tDB, codes, decodes)


""" A transaction database index storing covers of item in bitmaps """


class tDBIndex:
    """load transactions"""

    def __init__(self, tDB):
        items = set()
        for t in tDB:
            items |= set(t)
        covers = {item: pyroaring.BitMap() for item in items}
        for tid, t in enumerate(tDB):
            for item in t:
                covers[item].add(tid)
        self.covers = {s: pyroaring.FrozenBitMap(b) for s, b in covers.items()}
        self.ncolumns = len(items)
        self.nrows = len(tDB)

    """ return cover of an itemset/list of items """

    def cover(self, itemset, base=None):
        nitems = len(itemset)
        if nitems == 0:
            return pyroaring.BitMap(np.arange(self.nrows)) if base is None else base
        if base is None:
            return pyroaring.BitMap.intersection(
                *[self.covers[item] for item in itemset]
            )
        return pyroaring.BitMap.intersection(
            base, *[self.covers[item] for item in itemset]
        )

    """ return support of an itemset/list of items """

    def supp(self, itemset, base=None):
        return len(self.cover(itemset, base))


""" A contingency table. """


# contingency table
# =========== dec- ==== dec+ === tot
# protected    a         b       n1()
# unprotect.   c         d       n2()
# =========  m1()  ===  m2() === n()
class ContingencyTable:
    def __init__(self, a, n1, c, n2, avg_neg=0.5, ctx=None, protected=None):
        self.a = a
        self.b = n1 - a
        self.c = c
        self.d = n2 - c
        self.avg_neg = avg_neg
        self.ctx = ctx
        self.protected = protected

    def n1(self):
        return self.a + self.b

    def n2(self):
        return self.c + self.d

    def n(self):
        return self.a + self.b + self.c + self.d

    def m1(self):
        return self.a + self.c

    def m2(self):
        return self.b + self.d

    def __lt__(self, other):
        return self.ctx < other.ctx

    def __eq__(self, other):
        return self.ctx == other.ctx

    def __hash__(self):
        return hash(self.ctx)

    def p1(self):
        n1 = self.n1()
        if n1 > 0:
            return self.a / n1
        return self.avg_neg

    def p2(self):
        n2 = self.n2()
        if n2 > 0:
            return self.c / n2
        return self.avg_neg

    def p(self):
        return self.m1() / self.n()

    def rd(self):
        return self.p1() - self.p2()

    def ed(self):
        return self.p1() - self.p()

    def rr(self):
        p2 = self.p2()
        if p2 == 0:
            p2 = 1 / (self.n2() + 1)
        return self.p1() / p2

    def er(self):
        p = self.p()
        if p == 0:
            return 0
        return self.p1() / p

    def rc(self):
        p2 = self.p2()
        if p2 == 1:
            p2 = self.n2() / (self.n2() + 1)
        return (1 - self.p1()) / (1 - p2)

    def ec(self):
        p = self.p()
        if p == 1:
            n = self.n()
            p = n / (n + 1)
        return (1 - self.p1()) / (1 - p)

    def orisk(self):
        p1 = self.p1()
        if p1 == 1:
            p1 = self.a / (self.a + 1)
        p2 = self.p2()
        if p2 == 0:
            p2 = 1 / (self.n2() + 1)
        return p1 / (1 - p1) * (1 - p2) / p2


""" Minimum risk difference on a contingency table """


def check_rd(ctg, minSupp=20, threshold=0.1):
    # at least 20 protected with negative decision and 20 unprotected in total
    if ctg.a < 20 or ctg.n2() < 20:
        return None
    v = ctg.rd()
    # risk difference greater than 0.1
    return v if v > 0.1 else None


""" Discrimination discovery class. """


class DD:
    def __init__(
        self, df, unprotectedDesc, negdecDesc, na_values=None, one_hot_set=None
    ):
        if isinstance(df, pd.DataFrame):
            na_values = "NaN" if na_values is None else na_values
            self.tDB, self.codes, self.decodes = PD2tranDB(
                df, na_values=na_values, one_hot_set=one_hot_set
            )
        else:
            na_values = "?" if na_values is None else na_values
            self.tDB, self.codes, self.decodes = CSV2tranDB(
                df, na_values=na_values, one_hot_set=one_hot_set
            )
        self.unprotectedDesc = unprotectedDesc
        self.negdecDesc = negdecDesc
        self.sensitiveAtt = get_att(unprotectedDesc)
        self.decisionAtt = get_att(negdecDesc)
        self.unprotected = self.codes[unprotectedDesc]
        self.protected = [
            self.codes[v]
            for v in self.codes
            if get_att(v) == self.sensitiveAtt and self.codes[v] != self.unprotected
        ]
        self.neg_dec = self.codes[negdecDesc]
        pos_decs = [
            self.codes[v]
            for v in self.codes
            if get_att(v) == self.decisionAtt and self.codes[v] != self.neg_dec
        ]
        if len(pos_decs) != 1:
            raise ("binary decisions only!")
        self.pos_dec = pos_decs[0]
        self.posdecDesc = self.decodes[self.pos_dec]
        self.itDB = tDBIndex(self.tDB)
        self.unprCover = self.itDB.covers[self.unprotected]
        self.negCover = self.itDB.covers[self.neg_dec]
        self.avg_neg = len(self.negCover) / self.itDB.nrows

    def extract(self, testCond=lambda x: True, minSupp=20, target="c", maxn=0):
        exclude = {
            self.codes[v]
            for v in self.codes
            if get_att(v) in {self.sensitiveAtt, self.decisionAtt}
        }
        tDBprojected = [list(set(t) - exclude) for t in self.tDB]
        fisets = fim.fpgrowth(tDBprojected, supp=minSupp, zmin=0, target=target)
        q = Q.PriorityQueue()
        for fi in fisets:
            base = self.itDB.cover(fi[0])
            n2 = base.intersection_cardinality(self.unprCover)
            base_neg = base & self.negCover
            c = base_neg.intersection_cardinality(self.unprCover)
            for protected in self.protected:
                prCover = self.itDB.covers[protected]
                n1 = base.intersection_cardinality(prCover)
                a = base_neg.intersection_cardinality(prCover)
                ctg = ContingencyTable(a, n1, c, n2, self.avg_neg)
                v = testCond(ctg)
                if v is not None:
                    ctg.ctx, ctg.protected = fi[0], protected  # set only if test pass
                    q.put((v, ctg))
                    if q.qsize() > maxn:
                        q.get()
        return sorted([x for x in q.queue], reverse=True)

    def print(self, ctg):
        protectedDesc = self.decodes[ctg.protected]
        n = ctg.n()
        xlen = max(len(protectedDesc), len(self.unprotectedDesc))
        print("-----\nB =", " AND ".join([self.decodes[it] for it in ctg.ctx]))
        spec = (
            "{:"
            + str(xlen)
            + "}|{:"
            + str(len(self.negdecDesc))
            + "}|{:"
            + str(len(self.posdecDesc))
            + "}|{:"
            + str(len(str(n)))
            + "}"
        )
        print(spec.format("", self.negdecDesc, self.posdecDesc, ""))
        print(spec.format(protectedDesc, ctg.a, ctg.b, ctg.n1()))
        print(spec.format(self.unprotectedDesc, ctg.c, ctg.d, ctg.n2()))
        print(spec.format("", ctg.m1(), ctg.m2(), n))


"""
Sample usage 
if __name__ == '__main__':
    import time
    
    start_time=time.perf_counter() 
    
    disc = DD("credit.csv", 'foreign_worker=no', 'class=bad')
    #disc = DD("adult.csv", 'sex=Male', 'class=-50K')
    ctgs = disc.extract(testCond=check_rd, minSupp=-20, maxn=100)
    for v, ctg in ctgs[:2]:
        disc.print(ctg)
    print('==================')

    elapsed_time=time.perf_counter()-start_time 
    print('Elapsed time (s): {:.2f}'.format(elapsed_time) )
    print('Contingency tables: {}'.format(len(ctgs)))
"""
