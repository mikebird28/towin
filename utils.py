import keras.backend as K
import functools
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import  OneHotEncoder
import pandas as pd
import numpy as np

def process_features():
    PROCESS_FEATURE = [
         "hi_RaceID","ri_Year","hr_OrderOfFinish","hr_Distance","li_WinOdds","pops_order","margin",
         "hr_FinishingTime","hr_TimeDelta","margin","norm_time","div_time","is_win","is_place","div_margin","norm_margin",
         "hr_PaybackPlace","hr_PaybackWin","win_return","place_return","oof_over_pops","pop_prediction"
    ]
    return PROCESS_FEATURE

def generate_target_features(df,calc_pop = False):
    if calc_pop:
        df = calc_pop_order(df)
        df.loc[:,"oof_over_pops"] = (df["hr_OrderOfFinish"] < df["pops_order"]).astype(np.int32)
        df.loc[:,"pop_prediction"] = df["oof_over_pops"] * df["is_place"]

    #df.loc[df.loc[:,"hr_OrderOfFinish"] == 1, "hr_TimeDelta"] = 0.0
    df = calc_margin(df)

    df.loc[:,"is_win"] = (df.loc[:,"hr_OrderOfFinish"] == 1).astype(np.int32)
    df.loc[:,"is_place"] = df.loc[:,"hr_PaybackPlace"].notnull().astype(np.int32)
    df.loc[:,"win_return"] = (df.loc[:,"hr_PaybackWin"].fillna(0) - 100) / 100
    df.loc[:,"place_return"] = (df.loc[:,"hr_PaybackWin"].fillna(0) - 100) / 100
    df.loc[:,"norm_time"] = df.loc[:,"hr_FinishingTime"]/df.loc[:,"ri_Distance"]

    df = norm_with_distance(df,"hr_FinishingTime","norm_time")
    df = divide_with_distance(df,"hr_FinishingTime","div_time")
    return df

def calc_margin(df):
    #calculate margin
    tmp_key = "fastest"
    if tmp_key in df:
        raise Exception("temporl key already exists")
    df.loc[df["hr_FinishingTime"] == 0,"hr_FinishingTime"] = np.nan
    group = df.loc[:,["hi_RaceID","hr_FinishingTime"]].groupby("hi_RaceID").min()
    group.columns = [tmp_key]
    df = df.merge(group,left_on = "hi_RaceID",right_index = True)
    df.loc[:,"margin"] = df.loc[:,"hr_FinishingTime"] - df.loc[:,tmp_key]
    df.drop(tmp_key, axis = 1,inplace = True)

    #normalize margin with distance
    df = norm_with_distance(df,"margin","norm_margin")
    df = divide_with_distance(df,"margin","div_margin")
    return df

def norm_with_distance(df,target,name):
    dist_group = df.loc[:,["hr_Distance",target]].groupby("hr_Distance")
    dist_avg = dist_group.mean()
    dist_std = dist_group.std()
    dist_avg.columns = ["tmp_avg"]
    dist_std.columns = ["tmp_std"]
    dist_info = dist_avg.join(dist_std)

    df = df.merge(dist_info, on = "hr_Distance")
    df[name] = (df[target] - df["tmp_avg"])/df["tmp_std"]
    df.drop(["tmp_avg","tmp_std"], axis = 1, inplace = True)
    return df

def divide_with_distance(df,target,name):
    dist_group = df.loc[:,["hr_Distance",target]].groupby("hr_Distance")
    dist_avg = dist_group.mean()
    dist_std = dist_group.std()
    dist_avg.columns = ["tmp_avg"]
    dist_std.columns = ["tmp_std"]
    dist_info = dist_avg.join(dist_std)

    df = df.merge(dist_info, on = "hr_Distance")
    df[name] = df[target] / df["tmp_avg"]
    df.drop(["tmp_avg","tmp_std"], axis = 1, inplace = True)
    return df


def calc_pop_order(df):
    key = "li_WinOdds"
    if key not in df.columns:
        print("no key")
        return df
    df[key] = df.loc[:,key].where(df.loc[:,key] >= 1,99)
    df["pops_tmp"] = 1/df.loc[:,key]
    group = df.loc[:,["hi_RaceID","pops_tmp"]].groupby("hi_RaceID")
    for n,g in tqdm(group):
        g.sort_values(by = "pops_tmp",ascending = False,inplace = True)
        g.insert(0, 'pops_order', range(1,len(g)+1))
        df.loc[g.index,"pops_order"] = g.loc[:,"pops_order"]
    df.drop("pops_tmp",axis = 1,inplace = True)
    return df

def load_dtypes(dtypes_path):
    dtypes = {}
    allow = ["hr_OrderOfFinish","hr_TimeDelta","hr_PaybackWin","hr_PaybackPlace","hr_FinishingTime","hr_Distance"]
    with open(dtypes_path) as fp:
        for row in fp.readlines():
            row_s = row.strip().split(",")
            name = row_s[0]
            dtyp = row_s[1]
            if name.startswith("hr_") and (name not in allow):
                continue
            dtypes[name] = dtyp
    return dtypes

def load_removes(remove_path):
    ls = []
    with open(remove_path) as fp:
        for row in fp.readlines():
            row = row.strip()
            if row == "" or row.startswith("#"):
                continue
            ls.append(row.strip())
    return sorted(ls)

def load_targets(remove_path):
    ls = []
    with open(remove_path) as fp:
        for row in fp.readlines():
            ls.append(row.strip())
    return sorted(ls)

def cats_and_nums(usecols,dtypes):
    cats = []
    nums = []
    for k,v in dtypes.items():
        if v == "object":
            cats.append(k)
        else:
            nums.append(k)
    cats = sorted(cats)
    nums = sorted(nums)
    return (cats,nums)

def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


def replace_few(df,categoricals,threhold = 1000):
    remove = ["hi_RaceID"]
    categoricals = [c for c in df.columns if c in categoricals]
    categoricals = [c for c in categoricals if c not in remove]
    for c in tqdm(categoricals):
        tmp_name = c + "_vc"
        vc = df.loc[:,c].value_counts()
        vc.name = tmp_name
        vc = vc.to_frame()
        df = df.join(vc, on = c, how = "left")
        df[c] = df.loc[:,c].where(df.loc[:,tmp_name] > threhold, "others")
        df.drop(tmp_name,axis = 1, inplace = True)
    return df

def one_hot_encoding(df,categoricals):
    categoricals = [c for c in df.columns if c in categoricals]
    for c in categoricals:
        series = df.loc[:,c]
        encoded = OneHotEncoder().fit_transform(series)
        encoded = pd.DataFrame(encoded)
        df.drop(c,axis = 1,inplace = True)
        df = df.concat(encoded,axis = 1)
    return df
        


