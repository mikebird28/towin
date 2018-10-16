import preprep
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense,Activation,Dropout,Input,Conv2D,Concatenate,SpatialDropout1D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten,Lambda,Reshape
from keras.layers.embeddings import Embedding
import gc
from keras import backend as K
import utils
import logging
import tensorflow as tf
import add_features

PROCESS_FEATURES = utils.process_features()

RACE_FEATURES = ["hi_Distance"]
DTYPE_PATH = "./data/dtypes.csv"
TARGET_PATH = "./target.csv"
REMOVE_PATH = "./remove.csv"
DATASET_PATH = "./data/output.csv"

def main():
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(level = logging.DEBUG, filename = "cnn_exec.log", format = fmt)

    columns_dict = utils.load_dtypes(DTYPE_PATH)
    columns = sorted(columns_dict.keys())
    
    categoricals,numericals = utils.cats_and_nums(columns,columns_dict)
    categoricals = categoricals + add_features.additional_categoricals()
    removes = utils.load_removes("remove.csv")
    years = range(2011,2018+1)
    df = pd.read_csv("./data/output.csv",dtype = columns_dict,usecols = columns)

    p = preprep.Preprep("./cache_files/cnn")
    p = p.add(preprocess_target, name = "target_pre")
    p = p.add(fillna,params = {"categoricals":categoricals},name = "fillna")
    p = p.add(label_encoding,params = {"categoricals":categoricals}, name = "lenc", cache_format = "feather")
    p = p.add(choose_columns,params = {"removes" : removes,"process_features" : PROCESS_FEATURES}, name = "remove", cache_format = "feather")
    p = p.add(choose_year,params = {"years" : years}, name = "years", cache_format = "feather")
    p = p.add(normalize, params = {"process_features" : PROCESS_FEATURES, "categoricals" : categoricals}, name = "norm", cache_format = "feather")
    df = p.fit_gene(df,verbose = True)

    key = "hr_OrderOfFinish"
    y = df["hr_OrderOfFinish"]
    x = df.drop(key,axis = 1)
    x1,y1,x2,y2 = split_with_year(x,y,year = 2013)

    del(p,df);gc.collect()
    x1 = to_dict(x1)
    x2 = to_dict(x2)
    x1 = filter_df(x1)
    x2 = filter_df(x2)
    nn(x1,x2,y1,y2)

def nn(x1,x2,y1,y2):
    #y1 = np.log(y1 + 1)
    #y2 = np.log(y2 + 1)
    features = sorted(x1.keys())

    inputs = []
    flatten_layers = []
    max_dict = {}
    for f in x1.keys():
        if f == "main":
            continue
        max_dict[f] = int(max(x1[f].max().max(), x2[f].max().max()))

    for f in features:
        if f == "main":
            dim_1 = x1[f].shape[1]
            dim_2 = x1[f].shape[2]
            x = Input(shape = (dim_1,dim_2) ,dtype = "float32", name = f)
            inputs.append(x)
            flatten_layers.append(x)
            flatten_layers = x
        else:
            dim_1 = x1[f].shape[1]
            x = Input(shape = (dim_1,),dtype = "float32",name = f)
            inputs.append(x)
            inp_dim = max_dict[f] + 1
            out_dim = int(emb_dim(inp_dim))
            x = Embedding(inp_dim,out_dim,input_length = dim_1, embeddings_regularizer = keras.regularizers.l2(0.01))(x)
            x = SpatialDropout1D(0.2)(x)
            flatten_layers.append(x)

    x = Concatenate()(flatten_layers)
    x = flatten_layers
    feature_size = x.shape[2].value
    x = Reshape([18,feature_size,1])(x)

    layer_num = 3
    layer_size = 512
    bn_axis = -1
    momentum = 0.0
    dropout_rate = 0.2
    activation = "relu"

    x = BatchNormalization(axis = bn_axis,momentum = momentum)(x)

    x = Dense(units = 1024)(x)
    x = Activation(activation)(x)
    x = Dense(units = 18)(x)
    x = Activation("softmax")(x)

    model = Model(inputs = inputs,outputs = x)

    early_stopping = EarlyStopping(monitor='val_acc', patience= 50, verbose=0)
    check_point = ModelCheckpoint("./models/pair_nn.hd", period = 1, verbose = 0, save_best_only = True)

    #model.compile(loss = rank_sigmoid,optimizer="adam", metrics = [rank_accuracy])
    model.compile(loss='mse', optimizer='adam', metrics=[rank_accuracy])

    #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x1,y1,epochs = 50,batch_size = 32,validation_data = (x2,y2),callbacks = [early_stopping,check_point])
    print(model.predict(x2)[0])
    return model

def to_dict(df,categoricals):
    dic = {}
    numericals = []
    dic["main"] = to_matrix(df.loc[:,numericals])
    for c in categoricals:
        if c in df.columns:
            dic[c] = df[c]
    return dic

def emb_dim(inp_dim):
    if inp_dim < 5:
        return inp_dim
    elif inp_dim < 20:
        return 10
    else:
        return 20

def fillna(df,categoricals = [],mode = "race", logger = None):
    logger = logger or logging.getLogger(__name__)
    logger.debug("fillna : function called")

    df[df.isnull()] = np.nan
    df.dropna(subset = ["hr_OrderOfFinish"], inplace = True)

    nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
    logger.debug("nan rate check at start of fillna : {}".format(nan_rate))

    if mode == "mean":
        for f in tqdm(df.columns):
            if f in categoricals:
                df.loc[:,f].fillna("Nan",inplace = True)
            else:
                mean = df[f].mean()
                if np.isnan(mean):
                    mean = 0
                df.loc[:,f].fillna(mean,inplace = True)
        return df

    elif mode == "race":
        df.loc[:,categoricals].fillna("Nan",inplace = True)

        numericals = [c for c in df.columns if c not in categoricals]
        targets = numericals + ["hi_RaceID"]
        groups = df.loc[:,targets].groupby("hi_RaceID")
        group_mean = groups.mean().reset_index()
        group_mean.fillna(0,inplace = True)

        del(groups);gc.collect()

        mean_columns = []
        for c in group_mean.columns:
            if c  == "hi_RaceID":
                mean_columns.append(c)
            else:
                mean_columns.append("{}_mean".format(c))
        group_mean.columns = mean_columns
        df = df.merge(group_mean,on = "hi_RaceID", how = "left")

        for c in targets:
            if c == "hi_RaceID" or c in PROCESS_FEATURES:
                continue
            mean_col = "{}_mean".format(c)
            if mean_col not in df.columns:
                continue
            df.loc[:,c] = df.loc[:,c].fillna(df.loc[:,mean_col])

        drop_targets = [c for c in mean_columns if c != "hi_RaceID"]
        df.drop(drop_targets,inplace = True,axis = 1)

    nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
    logger.debug("nan rate check at end of fillna : {}".format(nan_rate))
    return df

def normalize(df,categoricals = [], process_features = None, logger = None):
    logger = logger or logging.getLogger(__name__)
    logger.debug("normalize : function called")

    remove = process_features
    nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
    logger.debug("nan rate befort normalization : {}".format(nan_rate))

    for k in df.columns:
        if k in remove:
            print(k)
            continue
        if k not in categoricals:
            std = df[k].std()
            mean = df[k].mean()
            if np.isnan(std) or std == 0:
                logger.debug("std of {} is 0".format(k))
                df.loc[:,k] = 0
            else:
                df.loc[:,k] = (df.loc[:,k] - mean)/std
    nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
    logger.debug("nan rate after normalization : {}".format(nan_rate))
    df = df.fillna(0)
    return df

def feature_engineering(df,categoricals):
    print("add speed")
    df = add_features.add_speed(df)

    print("add corner")
    df = add_features.add_corner_info(df)

    print("fix_extra_info")
    df = add_features.fix_extra_info(df)

    print("add_delta_info")
    df = add_features.add_delta_info(df)

    print("normalizing with race")
    df = add_features.norm_with_race(df,categoricals)
    return df

def normalize_with_race(df,numericals = []):
    key = "hi_RaceID"
    groups = df.groupby(key)
    for i,group in groups:
        pass
    for k in numericals:
        df[k] = (df - df[k].mean())/df[k].std()
    return df

def label_encoding(df,categoricals = [],logger = None):
    #init logger
    logger = logger or logging.getLogger(__name__)
    logger.debug("label_encoding : function called")

    dont_remove = ["hi_RaceID"]
    categoricals = extract_columns(df,categoricals)
    threhold = 20
    for k in tqdm(categoricals):
        #remove the few number of occurrences of  value
        if k not in dont_remove:
            bef_unum = len(df.loc[:,k].unique())
            v_counts = df.loc[:,k].value_counts()
            idx = df.loc[:,k].isin(v_counts.index[v_counts<threhold])
            df.loc[idx,k] = "few"
            aft_unum = len(df.loc[:,k].unique())
            logging.debug("label_encoding : encoding {},  unique value num  changed {} => {}".format(k,bef_unum,aft_unum))
        else:
            logging.debug("label_encoding : encoding {}".format(k))

        #label encoding
        le = LabelEncoder()
        df.loc[:,k] = le.fit_transform(df[k].astype(str))

    df = optimize_type(df,categoricals)
    return df

def choose_columns(df,removes = None,targets = None, process_features = []):
    if removes != None and targets != None:
        raise Exception("you can use only either of removes or taget")
    elif removes != None:
        columns = sorted(list(filter(lambda x : x not in removes, df.columns)))
        return df.loc[:,columns]
    elif targets != None:
        targets = sorted(list(set(targets + process_features)))
        columns = extract_columns(df,targets)
        return df.loc[:,columns]
    else:
        return df

def choose_year(df,years = None):
    year_column = "ri_Year"
    if years is None:
        return df
    elif year_column in df.columns:
        return df.loc[df.loc[:,year_column].isin(years),:]
    else:
        return df


def preprocess_target(df,logger = None):
    logger = logger or logging.getLogger(__name__)
    logger.debug("preprocess_target : function called")

    #drop rows if row doesn't have proces feature
    before_size = len(df)
    df.dropna(subset = ["hi_RaceID","hr_OrderOfFinish"],inplace = True)
    df = df[df["hr_OrderOfFinish"] != 0]
    after_size = len(df)
    logger.debug("preprocess_target : drop irrlegal, {} => {}".format(before_size,after_size))

    print("add target variables")
    df = utils.generate_target_features(df)

    #drop newbies
    before_size = len(df)
    df = df.loc[~df.loc[:,"hi_ConditionClass"].isin(["0s","s"]),:]
    after_size = len(df)
    logger.debug("preprocess_target : drop newbies, {} => {}".format(before_size,after_size))

    df = add_features.parse_feature(df)

    print("add_change_info")
    df = add_features.add_change_info(df)

    print("add_datetime_info")
    df = add_features.add_datetime_info(df)

    print("add_runstyle_info")
    df = add_features.add_run_style_info(df)

    print("add_surface_info")
    df = add_features.add_course_info(df)

    print("add_comb_info")
    df = add_features.add_combinational_feature(df)
    return df

def add_speed(df):
    targets = ["pre1","pre2","pre3"]
    for t in targets:
        speed = t + "_Speed"
        finishing_time = t + "_" + "FinishingTime"
        distance = t + "_" + "Distance"
        if (not finishing_time in df.columns) or (not distance in df.columns):
            continue
        df[speed] = df[finishing_time]/df[distance]

    for t in targets:
        oof_ratio = t + "_OofRatio"
        oof = t + "_OrderOfFinish"
        head_count = t + "_HeadCount"
        if (not oof in df.columns) or (not head_count in df.columns):
            continue
        df[oof_ratio] = df[oof]/df[head_count]
    return df

def add_race_avg(df,numericals):
    key = ["hi_RaceID"]
    numericals = [c for c in df.columns if c in numericals]
    targets = numericals + [k for k in key if k not in numericals]
    group = df[targets].groupby(key)
    group_mean = group.mean().reset_index()

    new_columns = []
    for c in group_mean.columns:
        if c in key:
            new_columns.append(c)
        else:
            new_columns.append("{}_mean".format(c))
    group_mean.columns = new_columns
    df = df.merge(group_mean,on = key, how = "left")
    return df

def filter_df(df):
    remove = ["hr_OrderOfFinish","hi_RaceID","ri_Year"] + PROCESS_FEATURES
    keys = [k for k in df.keys()]
    for k in keys:
        if k == "main":
            if k in remove:
                df[k] = df.drop(k)
        else:
            if k in remove:
                del(df[k])
    return df

def downsample(df):
    target = "hr_OrderOfFinish"
    df_true = df[df[target] == 1]
    df_false = df[df[target] == 0].sample(n = len(df_true),random_state = 32)
    df = pd.concat([df_true,df_false],axis = 0).sample(frac = 1.0)
    return df

def evaluate(df,y,model,features):
    df = pd.concat([df,y],axis = 1)
    df["pred"] = model.predict(filter_df(df[features]))
    is_hit = df[["hi_RaceID","pred","hr_OrderOfFinish"]].groupby("hi_RaceID").apply(_is_win)
    score = sum(is_hit)/len(is_hit)
    print(score)

def _is_win(df):
    is_hit = df["pred"].idxmax() == df["hr_OrderOfFinish"].idxmax()
    return is_hit

def split_with_year(x,y,year):
    con = pd.concat([x,y],axis = 1)
    train = con[con["ri_Year"] < year]
    test = con[con["ri_Year"] >= year]
    train.drop("ri_Year",axis = 1,inplace = True)
    test.drop("ri_Year",axis = 1,inplace = True)

    del(con)
    gc.collect()
    return (train.loc[:,x.columns],test.loc[:,x.columns],train.loc[:,y.columns],test.loc[:,y.columns])

def optimize_type(df,categoricals):
    for c in df.columns:
        if c not in categoricals:
            continue

        max_value = df[c].max()
        if max_value > 2^16 - 1:
            typ = np.int32
        else:
            typ = np.int16
        df[c] = df[c].astype(typ)
    return df

def extract_columns(df,columns):
    ls = [c for c in df.columns if c in columns]
    return ls

def to_matrix(df):
    #print(len(df.index.get_level_values(0)))
    dim1 = len(df.index.get_level_values(0).unique())
    dim2 = len(df.index.get_level_values(1).unique())
    result = df.values.reshape((dim1, dim2, df.shape[1]))
    return result

def check_dataframe(df,categoricals):
    for f in df.columns:
        if f in categoricals:
            typ = "categorical"
        else:
            typ = "numerical"
        mean = df.loc[:,f].mean()
        std = df.loc[:,f].std()
        txt = "f ({}) - mean : {}, std : {}".format(typ,mean,std)
        print(txt)

if __name__ == "__main__":
    main()
