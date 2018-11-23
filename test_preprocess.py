
import unittest
import preprocess
import utils
import pandas as pd
import numpy as np

DTYPE_PATH = "./data/dtypes.csv"
DATA_PATH = "./data/output.csv"

dataframe_mock = None
categoricals_mock = None

def load_dataset():
    columns_dict = utils.load_dtypes(DTYPE_PATH)
    columns = sorted(columns_dict.keys())
    categoricals,numericals = utils.cats_and_nums(columns,columns_dict)
    df = pd.read_csv("./data/output.csv",dtype = columns_dict,usecols = columns)
    df = df[df["ri_Year"] >= 2015]
    return df,categoricals

def get_cache():
    global dataframe_mock,categoricals_mock
    dataframe = dataframe_mock.copy()
    categoricals = [c for c in categoricals_mock]
    return dataframe,categoricals

dataframe_mock,categoricals_mock = load_dataset()

class RaceTestCase(unittest.TestCase):

    def test_fillna_horse(self):
        df,categoricals = get_cache()

        fillna = preprocess.fillna()
        numericals = [c for c in df.columns if c not in categoricals]
        targets = numericals + ["hi_RaceID"]

        group_mean_bef= df.loc[:,targets].groupby("hi_RaceID")[numericals].mean()
        print(group_mean_bef)

        df = fillna.on_fit(df,categoricals)
        nan_rate = 1 - df.isnull().sum().sum() / float(df.size)
        self.assertEqual(nan_rate,1)

        group_mean_new= df.loc[:,targets].groupby("hi_RaceID")[numericals].mean()
        np.testing.assertEqual(group_mean_bef.values,group_mean_new.values)

        norm = preprocess.normalize()
        df = norm.on_fit(df,categoricals)


if __name__ == "__main__":
    unittest.main()
