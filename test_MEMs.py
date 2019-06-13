from unittest import TestCase
import pandas as pd
from numpy.random import randint
import numpy as np
from mem import MEMs

class mod():
    def predict(self,row):
        """
        a dummy model with a predict function
        :param val: assumes single row dataframe
        :return:
        """
        return row.sum(axis=1)[0]*2

class TestgetMem(TestCase):
    def setUp(self):
        pass


class TestMEMs(TestCase):
    def setUp(self):
        self.dfo_binary = pd.DataFrame([[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
                                          columns=list('abc'))
        self.dfe_binary = pd.DataFrame([[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]],
                                       columns=list('abc'))
        self.dfe_binary1 = pd.DataFrame([[0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]],
                                      columns=list('abc'))

        self.dfo_scrambled = pd.DataFrame([[9,10,11],[6,7,8],[3,4,5],[12,13,14],[0,1,2]], columns=list('abc'))
        self.dfe_scrambled = pd.DataFrame([[9, 10, 11], [6, 7, 8], [12, 13, 14], [0, 1, 2]],columns=list('abc'))

        self.df111 = pd.DataFrame([[0,0,1],[1,11,5],[2,22,19]], columns=list('abc'))
        self.df1 = pd.DataFrame(np.arange(12).reshape(4, 3), columns=list('abc'))
        self.df2 = pd.DataFrame(np.arange(15).reshape(5, 3), columns=list('abc'))
        self.df11 = pd.DataFrame(np.arange(60).reshape(20, 3), columns=list('abc'))
        self.df22 = pd.DataFrame(np.arange(63).reshape(21, 3), columns=list('abc'))
        self.col='a'
        self.col0=0#
        self.df1f = pd.DataFrame(np.linspace(1.0, 10.0,12).reshape(4, 3), columns=list('abc'))
        self.df2f = pd.DataFrame(np.linspace(1.0, 10.0,15).reshape(5, 3), columns=list('abc'))
        self.df11f = pd.DataFrame(np.linspace(1.0, 10.0,60).reshape(20, 3), columns=list('abc'))
        self.df22f = pd.DataFrame(np.linspace(1.0, 10.0,63).reshape(21, 3), columns=list('abc'))

    def test_getMEM_avgplusone(self):
        m = mod()


        # small even
        dfc = MEMs(self.dfo_binary)
        preds = dfc.getMEM_avgplusone(m, 'b')
        self.assertTrue(preds==[(1,6.0),(0,4.0)])

        dfc = MEMs(self.dfe_binary)
        preds = dfc.getMEM_avgplusone(m, 'b')
        self.assertTrue(preds == [(0, 0.0), (1, 2.0)])

        dfc = MEMs(self.dfo_scrambled)
        preds = dfc.getMEM_avgplusone(m, 'b')
        self.assertTrue(preds == [(7, 42.0), (8, 44.0)])

        dfc = MEMs(self.dfe_scrambled)
        preds = dfc.getMEM_avgplusone(m, 'b')
        self.assertTrue(preds == [(7, 42.0), (8, 44.0)])

    def test_handle_even_int_column_binary_choice(self):
        # small even
        dfc = MEMs(self.dfo_binary)
        self.assertTrue(dfc.df_avg.iloc[0, 0] == 1)
        dfc = MEMs(self.dfe_binary)
        self.assertTrue(dfc.df_avg.iloc[0, 0] == 0)
        dfc = MEMs(self.dfe_binary1)
        self.assertTrue(dfc.df_avg.iloc[0, 0] == 0)

    def test_handle_even_int_column_mean_choice(self):
        # small even
        dfc = MEMs(self.df111)
        a=dfc.df_avg.iloc[0, 0]
        self.assertTrue(dfc.df_avg.iloc[0,0]==1)
        self.assertTrue(dfc.df_avg.iloc[0, 1] == 11)
        self.assertTrue(dfc.df_avg.iloc[0, 2] == 5)

    def test_getMEM_small(self):
        # small even
        dfc = MEMs(self.df1)
        m=mod()
        preds = dfc.getMEM(m,'b')
        self.assertTrue(preds== [(1, 18.0), (4, 24.0), (7, 30.0), (10, 36.0)])
        self.assertRaises(KeyError, dfc._getRangeList, 'A', 10) #A does not exist
        self.assertRaises(KeyError, dfc._getRangeList, 'd', 10) #d "
        preds = dfc.getMEM(m, 1)
        self.assertTrue(preds== [(1, 18.0), (4, 24.0), (7, 30.0), (10, 36.0)])

        # small odd
        dfc = MEMs(self.df2)
        m = mod()
        self.assertRaises(KeyError, dfc._getRangeList, 'A', 10) #A does not exist
        self.assertRaises(KeyError, dfc._getRangeList, 'd', 10) #d "
        preds = dfc.getMEM(m,'b')
        self.assertTrue(preds== [(1, 30.0), (4, 36.0), (7, 42.0), (10, 48.0),(13,54.0)])

        #scrambled
        dfc = MEMs(self.dfo_scrambled)
        m=mod()
        preds = dfc.getMEM(m,'b')
        self.assertTrue(preds== [(1, 30.0), (4, 36.0), (7, 42.0), (10, 48.0),(13,54.0)])

        dfc = MEMs(self.dfe_scrambled)
        m = mod()
        preds = dfc.getMEM(m, 'b')
        self.assertTrue(preds == [(1, 30.0), (7, 42.0), (10, 48.0), (13, 54.0)])


    def test_getMEM_large(self):
        dfc = MEMs(self.df1)
        m = mod()
        self.assertRaises(KeyError, dfc._getRangeList, 'A', 10)  # A does not exist
        self.assertRaises(KeyError, dfc._getRangeList, 'd', 10)  # d "

        preds = dfc.getMEM(m, 'b')
        self.assertTrue(preds == [(1, 18.0), (4, 24.0), (7, 30.0), (10, 36.0)])
        preds = dfc.getMEM(m, 1)
        self.assertTrue(preds == [(1, 18.0), (4, 24.0), (7, 30.0), (10, 36.0)])
        pass  # get first column

    def test__getRangeList_ieven(self):
        sf = MEMs(self.df1)
        l= sf._getRangeList(0,10)
        self.assertTrue( (l==[0,3,6,9]).all() )
        l= sf._getRangeList(self.col,10)
        self.assertTrue( (l==[0,3,6,9]).all() )

    def test__getRangeList_iodd(self):
        sf = MEMs(self.df2)
        l= sf._getRangeList(0,10)
        self.assertTrue( (l==[0,3,6,9,12]).all() )
        l = sf._getRangeList(self.col, 10)
        self.assertTrue((l == [0, 3, 6, 9, 12]).all())
    def test__getRangeList_ieven_lrg(self):
        sf = MEMs(self.df11)
        l= sf._getRangeList(0,10)
        self.assertTrue( (l==[0,6,12,18,24,30,36,42,48,57]).all() )
        l= sf._getRangeList(self.col,10)
        self.assertTrue( (l==[0,6,12,18,24,30,36,42,48,57]).all() )
        l = sf._getRangeList(self.col, 3)
        self.assertTrue((l == [0, 30, 57]).all())
    def test__getRangeList_iodd_lrg(self):
        sf = MEMs(self.df22)
        l= sf._getRangeList(0,10)
        self.assertTrue( (l==[0,6,12,18,24,30,36,42,48,60]).all() )
        l = sf._getRangeList(self.col, 10)
        self.assertTrue((l ==[0,6,12,18,24,30,36,42,48,60]).all())
        l = sf._getRangeList(self.col, 3)
        self.assertTrue((l ==[0,30,60]).all())

    def check1(self, l, expectedvals):
        self.assertTrue(len(l) == expectedvals)  # make sure there are 4
        self.assertTrue(all(sorted(l) == l))  # make sure its sorted
        self.assertTrue(all(l[i] <= l[i + 1] for i in range(len(l) - 1)))  # make sure they are different

    def test__getRangeList_feven(self):
        sf = MEMs(self.df1f)
        l = sf._getRangeList(0, 10)
        self.check1(l,4)
        l = sf._getRangeList(self.col, 10)
        self.check1(l,4)
        l = sf._getRangeList(self.col, 3)
        self.check1(l,3)
        l = sf._getRangeList(self.col, 2)
        self.check1(l, 2)

    def test__getRangeListf_fodd(self):
        sf = MEMs(self.df2f)
        l = sf._getRangeList(0, 10)
        self.check1(l, 5)
        l = sf._getRangeList(self.col, 10)
        self.check1(l, 5)
        l = sf._getRangeList(self.col, 5)
        self.check1(l, 5)
        l = sf._getRangeList(self.col, 3)
        self.check1(l, 3)
        l = sf._getRangeList(self.col, 2)
        self.check1(l, 2)

    def test__getRangeList_feven_lrg(self):
        sf = MEMs(self.df11f)
        l = sf._getRangeList(0, 10)
        self.check1(l, 10)
        l = sf._getRangeList(self.col, 10)
        self.check1(l, 10)
        l = sf._getRangeList(self.col, 5)
        self.check1(l, 5)
        l = sf._getRangeList(self.col, 3)
        self.check1(l, 3)
        l = sf._getRangeList(self.col, 2)
        self.check1(l, 2)


    def test__getRangeList_fodd_lrg(self):
        sf = MEMs(self.df22f)
        l = sf._getRangeList(0, 10)
        self.check1(l, 10)
        l = sf._getRangeList(self.col, 10)
        self.check1(l, 10)
        l = sf._getRangeList(self.col, 5)
        self.check1(l, 5)
        l = sf._getRangeList(self.col, 3)
        self.check1(l, 3)
        l = sf._getRangeList(self.col, 2)
        self.check1(l, 2)


