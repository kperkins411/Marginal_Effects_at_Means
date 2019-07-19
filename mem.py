import pandas as pd
from numpy.random import randint
import numpy as np


class MEMs():
    """
    Marginal Effects At Means - used to get average value for all columns and then permute 1 column to see what model predicts

    NOTE: For int64 columns the average of the column is the integer value closest to the average computed for the column
    For instance if column average=4.5 and the column contains a mix of 1,3,5,7,9 ints.   The also will select 5
    since thats the closest to 4.5

    makes copy of passed in dataframe,
    -Construct: creates a 1 row dataframe (df_avg) that contains the average for each column.
    -getMEM: For a particular column, generates predictions based on df_avg and the permuted column
    """

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def __init__(self, df, verbose=False):
        '''
        takes in a dataframe, creates a copy with single row, averages each df column and puts it in copys column
        :param df: dataframe, numeric
        '''
        self.df_orig = df
        self.df_avg = pd.DataFrame(data=None, columns=df.columns, index=[0])  # create empty dataframe with same columns
        self.verbose=verbose

        # average df, put value into appropriate column of df_avg
        # WHAT TO DO ABOUT CATEGORICAL WITH 2 OR 4 VALUES? WHAT IS THE MEAN?
        #         for col in self.df_avg.columns:
        #             if (self.df_orig[col].dtype == np.int64):
        #                 #categorical!, set average equal to closest int to the mean of the column
        #                 vals = df[col].unique()    #get unique values
        #                 mn = df[col].mean()        #get the mean
        #                 self.df_avg.at[0, col] = self.find_nearest(vals, mn)
        #             elif (self.df_orig[col].dtype == np.float64):
        #                 #float, average is the mean
        #                 self.df_avg.at[0,col] = df[col].mean()
        #             else:
        #                 raise TypeError(f"All input columns must be an integer or a float, column {col} is a {str(self.df_orig[col].dtype)}")

        for col in self.df_avg.columns:
            self.df_avg.at[0, col] = df[col].mean()

    def getMEM_avg(self, model):
        # get the average prediction
        return model.predict_proba(self.df_avg)


    def getMEM_avgplusoneSimple(self, model, col):
        """
        Gets predictions of avg + 1
        :param model: random forest
        :param col: which column to operate on, string (ex. "b") or index (ex. 0)
        :return: list of (col_val, prediction)
        finds range of column in self.df_orig, create list with number_steps going from range.start to range.end
        runs predictions on self.df_avg with those ranged values

        """
        # if its an int get the string column name
        if (type(col) is int):
            col = self.df_orig.columns[col]

        preds = []

        # get the average prediction
        df_avgtmp = self.df_avg.copy()

        # lets see if we are dealing with a binary value
        if self.df_orig[col].nunique() == 2:
            #if binary, predict min and then max value
            df_avgtmp.at[0, col] = self.df_orig[col].min()
            p1 = model.predict_proba(df_avgtmp)
            preds.append((df_avgtmp.at[0, col], p1[0, 1]))

            df_avgtmp.at[0, col] = self.df_orig[col].max()
            p2 = model.predict_proba(df_avgtmp)
            preds.append((df_avgtmp.at[0, col], p2[0, 1]))
            if (self.verbose): print(f'averageprob={p1} plus1prob={p2}')

        else:
            # continuous variable
            # assumme we can add 1
            add_amt = 1

            p1 = model.predict_proba(df_avgtmp)
            preds.append((df_avgtmp.at[0, col], p1[0, 1]))

            df_avgtmp.at[0, col] = df_avgtmp.at[0, col] + 1
            p2 = model.predict_proba(df_avgtmp)
            preds.append((df_avgtmp.at[0, col], p2[0, 1]))
            if (self.verbose): print(f'averageprob={p1} plus1prob={p2}')

        return preds

    def getMEM_avgplusoneSimple_Probability_Change(self, model, col):
        preds = self.getMEM_avgplusoneSimple(model, col)
        return preds[1][1] - preds[0][1]

    def getMEM(self, model, col, number_steps=10.0):
        """
        :param model: random forest
        :param col: which column to operate on, string (ex. "b") or index (ex. 0)
        :param number_steps: how many iterations
        :return: list of (col_val, prediction)
        finds range of column in self.df_orig, create list with number_steps going from range.start to range.end
        runs predictions on self.df_avg with those ranged values
        If column is categorical the vals selected are chosen from the available categories
        """

        vals = self._getRangeList(col, number_steps)
        preds = []

        # if its an int get the string column name
        if (type(col) is int):
            col = self.df_orig.columns[col]

        # run predictor for every value in vals over the average of the other columns
        for val in vals:
            df_avgtmp = self.df_avg.copy()
            df_avgtmp.at[0, col] = val
            preds.append((val, model.predict(df_avgtmp)))
        return preds

    def _getRangeList(self, col, number_samples):
        """
        gets number_samples starting at col.min() and ending at col.max()
        if column is categorical (int64) will be less if there are not number_samples unique values
        :param col: which column, string (ex. "b") or index (ex. 0)
        :param number_samples: how many you want,
        :return: list of values, uniformly distributed between start and finish
        """
        if (type(col) is int):
            vals = self.df_orig.iloc[:, col]
        else:
            vals = self.df_orig[col]

        unique_vals = np.sort(vals.unique())
        tpe = unique_vals.dtype
        luv = len(unique_vals)

        # choose smaller of the 2
        if (luv < number_samples):
            number_samples = luv

        res = []
        if (tpe == 'int64'):
            # its categorical
            step_size = int(luv / (number_samples - 1))
            res = unique_vals[0:luv:step_size]
            if (len(res) < number_samples):
                res = np.append(res, [-1])
            elif (len(res) > number_samples):
                res = res[:number_samples]
            res[len(res) - 1] = vals.max()
        elif (tpe == 'float64'):
            res = np.linspace(vals.min(), vals.max(), number_samples)

        return res
