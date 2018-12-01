
import preprep
import logging
from tqdm import tqdm
import numpy as np
import gc

#fillna functions
class fillna(preprep.Operator):
    def __init__(self):
        super().__init__()
        self.mean_dict = {}

    def on_fit(self,df,categoricals = [], logger = None):
        logger = logger or logging.getLogger(__name__)
        logger.debug("fillna : function called on fit")
        return self.process(df,categoricals = categoricals,mode = "fit",logger = logger)

    def on_pred(self,df,categoricals = [], logger = None):
        logger = logger or logging.getLogger(__name__)
        logger.debug("fillna : function called on fit")
        return self.process(df,categoricals = categoricals,mode = "pred",logger = logger)

    def process(self,df,mode = "fit",categoricals = [], logger = None):
        df[df.isnull()] = np.nan
        df.loc[:,"hr_PaybackWin_eval"] = df.loc[:,"hr_PaybackWin"].copy().fillna(0)/100
        df.loc[:,"hr_PaybackPlace_eval"] = df.loc[:,"hr_PaybackPlace"].copy().fillna(0)/100
        df.loc[:,"hr_PaybackWin"] = (df.loc[:,"hr_PaybackWin"].fillna(0)/100).clip(0.0,100.0)
        df.loc[:,"hr_PaybackPlace"] = (df.loc[:,"hr_PaybackPlace"].fillna(0)/100).clip(0.0,100.0)

        remove = ["hr_PaybackWin","hr_PaybackWin_eval","hr_PaybackPlace","hr_PaybackPlace_eval"]
        #df = _fillna_with_race(df,categoricals,remove = remove)
        _fillna_with_horse(df,categoricals,remove = remove)
        nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
        logger.debug("nan rate check at end of fillna : {}".format(nan_rate))
        return df

def _fillna_with_horse(df, categoricals, remove = None):
    key = "hi_RaceID"
    if remove is None:
        remove = [key]
    else:
        remove += key

    categoricals = [c for c in df.columns if c in categoricals]
    numericals = [c for c in df.columns if c not in categoricals + remove]

    to_float(df,numericals)
    for f in tqdm(numericals):
        mean = df[f].mean()
        if np.isnan(mean):
            mean = 0
        df.loc[:,f] = df.loc[:,f].fillna(mean)
    df.loc[:,categoricals] = df.loc[:,categoricals].fillna("nan")

def _fillna_with_race(df,categoricals, remove = None):
    key = "hi_RaceID"
    if remove is None:
        remove = [key]
    else:
        remove += key

    categoricals = [c for c in df.columns if c in categoricals]
    numericals = [c for c in df.columns if c not in categoricals + remove]
    targets = numericals + [key]

    #numericals
    to_float(df,numericals)

    mean = df.loc[:,targets].groupby(key).mean()
    mean_columns = []
    for c in mean.columns:
        if c == key:
            mean_columns.append(c)
            continue
        col_name = "{}_mean".format(c)
        mean_columns.append(col_name)
    mean.columns = mean_columns
    df = df.merge(mean,left_on = "hi_RaceID", right_index = True, how = "left")

    del(mean)
    gc.collect()

    to_float(df,numericals)
    for c in tqdm(targets):
        if c == key:
            continue
        else:
            mean_col = "{}_mean".format(c)
        df.loc[:,c] = df.loc[:,c].fillna(df.loc[:,mean_col])
        df.drop(mean_col, inplace = True, axis  = 1)

    df.loc[:,numericals] = df.loc[:,numericals].fillna(0)
    df.loc[:,categoricals] = df.loc[:,categoricals].fillna("nan")
    return df

class normalize(preprep.Operator):
    def __init__(self):
        self.mean_dict = {}
        self.std_dict = {}

    def on_fit(self,df,categoricals = [] ,logger = None):
        return self.normalize(df,"fit",categoricals,logger)

    def on_pred(self,df,categoricals = [] ,logger = None):
        return self.normalize(df,"pred",categoricals,logger)

    def normalize(self,df,mode,categoricals = [],logger = None):
        logger = logger or logging.getLogger(__name__)
        logger.debug("normalize : function called")

        #df["null_count_raw"] = df.loc[:,"null_count"].copy()
        #df["pre1_Distance_raw"] = df.loc[:,"pre1_Distance"].copy()
        #df["pre2_Distance_raw"] = df.loc[:,"pre2_Distance"].copy()
        #df["pre3_Distance_raw"] = df.loc[:,"pre3_Distance"].copy()
        race_features = [
            "ri_Distance","li_FieldStatus","hi_Times","ri_FirstPrize","ri_HaedCount","ri_Month","li_WinOdds",
            "nige_ratio","senko_ratio","sasi_ratio","oikomi_ratio","kouji_ratio","jizai_ratio","null_count_raw",
            "season_sin","season_cos","pre1_Distance_raw","pre2_Distance_raw","pre3_Distance_raw"
        ]
        binary_features = [
            "has_pre1","has_pre2","has_pre3","has_li","has_ei"
        ]
        race_features = [c for c in df.columns if c in race_features]
        binary_features = [c for c in df.columns if c in binary_features]

        remove = ["hr_OrderOfFinish","ri_Year","hi_RaceID","hr_PaybackWin","hr_PaybackPlace","hr_PaybackWin_eval","hr_PaybackPlace_eval"] + race_features + binary_features
        numericals = [c for c in df.columns if c not in set(categoricals + remove)]

        nan_rate = 1 - df.loc[:,numericals].isnull().sum().sum()/float(df.size)
        logger.debug("nan rate before normalization : {}".format(nan_rate))

        df = _norm_with_race(df,numericals)
        df = _norm_with_df(df,race_features)
        nan_rate = 1 - df.loc[:,numericals].isnull().sum().sum()/float(df.size)

        logger.debug("nan rate after normalization : {}".format(nan_rate))
        return df

def _norm_with_df(df,numericals = [], logger = None):
    for key in numericals:
        mean = df.loc[:,key].mean()
        std = df.loc[:,key].std()
        std = std.clip(1e-5,None)
        df.loc[:,key] = (df.loc[:,key] - mean)/std
    return df

def _norm_with_race(df,numericals = [],logger = None):
    key = "hi_RaceID"
    logger = logger or logging.getLogger(__name__)

    #drop duplicates

    numericals = sorted([c for c in list(set(numericals)) if c != key])
    targets = numericals + [key]

    groups = df.loc[:,targets].groupby(key)
    mean = groups.mean().reset_index()
    std = groups.std().reset_index()
    std.loc[:,numericals] = std.loc[:,numericals].clip(lower = 1e-5)

    mean_cols = []
    std_cols = []
    mean_prefix = "mean_"
    std_prefix = "std_"

    for c in mean.columns:
        if c == key:
            mean_name = c
        else:
            mean_name = mean_prefix+c
        mean_cols.append(mean_name)

    for c in std.columns:
        if c == key:
            std_name = c
        else:
            std_name = std_prefix+c
        std_cols.append(std_name)

    mean.columns = mean_cols
    std.columns = std_cols

    df = df.merge(mean,on = key, how = "left")
    df = df.merge(std,on = key, how = "left")

    for c in targets:
        if c == key:
            continue
        mean_name = mean_prefix + c
        std_name = std_prefix + c
        df.loc[:,c] = (df.loc[:,c] - df.loc[:,mean_name])/df.loc[:,std_name]

    drop_targets = [c for c in mean_cols + std_cols if c != key]
    df.drop(drop_targets,axis = 1, inplace = True)
    return df

def to_float(df,targets):
    for c in df.columns:
        if c not in targets:
            continue
        df.loc[:,c] = df.loc[:,c].astype(np.float32)

