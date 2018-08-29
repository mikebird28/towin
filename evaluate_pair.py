
import pandas as pd
import preprep
import functools
import tensorflow as tf
import numpy as np
from keras.models import load_model
import keras.backend as K
from utils import cats_and_nums, load_dtypes, load_removes , as_keras_metric

def main():
    df = pd.read_feather("cache_nn/norm.feather")
    df = df[df["ri_Year"] == 2017]
    df.drop(["index"],inplace = True, axis = 1)

    columns_dict = load_dtypes("dtypes.csv")
    columns = sorted(columns_dict.keys())
    categoricals,numericals = cats_and_nums(columns,columns_dict)

    auc_roc = as_keras_metric(tf.metrics.auc)
    model = load_model("./models/pair_nn.hd",custom_objects = {"auc":auc_roc})

    total = 0
    win = 0
    for i,grouped in df.groupby(["hi_RaceID"]):
        is_win = is_hit(grouped,model,categoricals)
        total += 1
        win += is_win
        #print(is_win)
        print(win/total)

def is_hit(df,model,categoricals):
    target = "hr_OrderOfFinish"
    y = df[target]
    df = df.reset_index(drop = True)
    sorted_idx = sort_with_pred(df,model,categoricals)
    df = df.iloc[sorted_idx,:].reset_index(drop = True)
    #print(df.loc[:,[target,"hi_BaseOdds"]])
    is_win = 0
    if df.loc[0,"hr_OrderOfFinish"] == 1:
        is_win = 1
    return is_win

def sort_with_pred(df,model,categoricals,left_priority = True,target = None):
    if target is None:
        target = [i for i in range(len(df))]

    if len(target) == 1:
        return target

    if len(target) % 2 == 1 and left_priority:
        mid = len(target)//2
    elif len(target) % 2 == 1 and not left_priority:
        mid = len(target)//2 + 1
    else:
        mid = len(target)//2

    next_priority = not left_priority
    left = sort_with_pred(df,model,categoricals,next_priority,target = target[:mid])
    right = sort_with_pred(df,model,categoricals,next_priority,target = target[mid:])
    size = len(left) + len(right)

    count = 0
    left_count = 0
    right_count = 0
    new_target = []

    while len(left) != left_count and len(right) != right_count:
        lidx = left[left_count]
        ridx = right[right_count]
        pred_df = df.iloc[[lidx,ridx],:]
        pred_df = to_predicable(pred_df,categoricals)
        pred = model.predict(pred_df)
        #case left is stronger

        if pred[0][0] > 0.5:
            new_target.append(lidx)
            left_count += 1
        #case right is stronger
        else:
            new_target.append(ridx)
            right_count += 1
        count += 1

    if len(left) == left_count:
        new_target.extend(right[right_count:])
    elif len(right) == right_count:
        new_target.extend(left[left_count:])
    return new_target

def to_predicable(df,categoricals):
    remove = ["hr_OrderOfFinish","ri_Year","hi_RaceID"]
    remove = ["hr_OrderOfFinish","hi_RaceID"]
    df = df.loc[:,~df.columns.isin(remove)]

    dic = {}
    main_mtx = df.loc[:,~df.columns.isin(categoricals)].values
    dic["main"] = main_mtx.reshape([1,2,main_mtx.shape[1]])
    for c in df.columns:
        if c not in categoricals:
            continue
        mtx = df.loc[:,[c]].values
        mtx = mtx.reshape([1,2])
        dic[c] = mtx
    return dic

if __name__ == "__main__":
    main()
