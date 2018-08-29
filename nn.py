import lightgbm as lgb
import preprep
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split,KFold
from skopt.space import Real,Integer
from skopt import gp_minimize
import keras
from keras.models import Sequential,Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense,Activation,Dropout,Input,Conv2D,Concatenate,SpatialDropout1D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.layers.embeddings import Embedding
import gc
import functools
from keras import backend as K
import tensorflow as tf


def main():
    columns_dict = load_dtypes("dtypes.csv")
    columns = sorted(columns_dict.keys())
    categoricals,numericals = cats_and_nums(columns,columns_dict)
    removes = load_removes("remove.csv")

    df = pd.read_csv("output.csv",dtype = columns_dict,usecols = columns)
    df = df.sort_index(axis=1, ascending=True)
    p = preprep.Preprep("./cache_nn")
    p = p.add(preprocess_target, name = "target_pre")
    p = p.add(fillna,params = {"categoricals":categoricals},name = "fillna")
    p = p.add(label_encoding,params = {"categoricals":categoricals}, name = "lenc", cache_format = "feather")
    p = p.add(remove_unused,params = {"removes" : removes}, name = "remove", cache_format = "feather")
    p = p.add(downsample,name = "downsample", cache_format = "feather")
    p = p.add(normalize, params = {"numericals" : numericals}, name = "norm", cache_format = "feather")
    p = p.add(inp_to_dict,params = {"categoricals" : categoricals}, name = "to_dict", cache_format = "pickle")
    x1,x2,y1,y2 = p.fit_gene(df,verbose = True)
    del(p);gc.collect()

    #df_ds,y_ds = downsample(df,y)
    x1 = filter_df(x1)
    x2 = filter_df(x2)
    #model = light_gbm(fdf,y,categoricals)
    model = nn(x1,x2,y1,y2)
    #optimize_params(fdf,y,categoricals)

    #features = list(filter(lambda x : x not in remove, x.columns))
    features = df.columns
    evaluate(x2,y2,model,features)

def search_unnecessary(fdf,categoricals):
    score = light_gbm(fdf,y,categoricals)
    for col in fdf.columns:
        feat_except_col = [c for c in fdf.columns if c != col]
        cat_except_col = [c for c in categoricals if c != col]
        s = light_gbm(fdf.loc[:,feat_except_col],y,cat_except_col,show_importance = False)
        typ = "T"
        if s < score:
            typ = "F"
        print("{} - score except  {} is  {}".format(typ,col,s))


def nn(x1,x2,y1,y2):
    features = sorted(x1.keys())

    inputs = []
    flatten_layers = []
    max_dict = {}
    for f in x1.keys():
        max_dict[f] = int(max(x1[f].max().max(), x2[f].max().max()))

    for f in features:
        if f == "main":
            shape = x1[f].shape[1]
            x = Input(shape = (shape,),dtype = "float32", name = f)
            inputs.append(x)
        else:
            x = Input(shape = (1,),dtype = "float32",name = f)
            inputs.append(x)
            inp_dim = max_dict[f] + 1
            out_dim = int(emb_dim(inp_dim))
            x = Embedding(inp_dim,out_dim,input_length = 1)(x)
            x = SpatialDropout1D(0.2)(x)
            x = Flatten()(x)
        flatten_layers.append(x)
    x = Concatenate()(flatten_layers)
    
    hidden_1 = 128
    x = Dense(units=hidden_1)(x)
    x = Activation("relu",name = "embedding")(x)
    x = BatchNormalization(momentum = 0.0)(x)
    x = Dropout(0.5)(x)

    x = Dense(units = 1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs = inputs,outputs = x)
    opt = keras.optimizers.Adam(lr=0.01,epsilon = 1e-2)
    auc_roc = as_keras_metric(tf.metrics.auc)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    model.compile(loss = "binary_crossentropy",optimizer=opt,metrics = [auc_roc])

    model.fit(x1,y1,epochs = 100,batch_size = 2000,validation_data = (x2,y2),callbacks = [early_stopping])
    return model

def emb_dim(inp_dim):
    if inp_dim < 10:
        return inp_dim
    elif inp_dim < 20:
        return 10
    else:
        return 20


def optimize_params(train_x,train_y,categoricals):
    features = train_x.columns
    categoricals = list(filter(lambda x: x in categoricals,features))
    metrics = "auc"
    def __objective(values):
        n_folds = 3
        num_boost_round = 20000
        early_stopping_rounds = 50
        print(values)
        params = {
            'max_depth': values[0],
            'num_leaves': values[1],
            'min_child_samples': values[2],
            'scale_pos_weight': values[3],
            'subsample': values[4],
            'colsample_bytree': values[5],
            'bagging_freq': 5,

            'learning_rate':0.01,
            #'subsample_freq': 1,
            'metric':metrics,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'task':'train',
            'verbose':-1,
        }

        kf = KFold(n = train_x.shape[0],n_folds = n_folds,random_state = 32,shuffle = True)
        scores = []
        for i,(train_index,test_index) in enumerate(kf):
            dtrain = lgb.Dataset(train_x.iloc[train_index], label=train_y.iloc[train_index],categorical_feature = categoricals)
            dtest  = lgb.Dataset(train_x.iloc[test_index],  label=train_y.iloc[test_index],categorical_feature = categoricals)
            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dtrain, dtest],
                valid_names=['train','valid'],
                evals_result={},
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False, feval=None
            )
            score = model.best_score["valid"][metrics]
            scores.append(score)
            print("[i] score ... {}".format(score))
        avg_score = -sum(scores)/len(scores)
        print("avg_score : {}".format(avg_score))
        print()
        return avg_score

    space = [
        Integer(5, 25, name='max_depth'),
        Integer(100, 700, name='num_leaves'),
        Integer(1, 2, name='min_child_samples'), #1
        Real(1, 7,  name='scale_pos_weight'), #1-7
        Real(0.3, 0.8, name='subsample'),
        Real(0.3, 0.8, name='colsample_bytree'),
    ]
    res_gp = gp_minimize(__objective, space, n_calls=30,random_state=32,n_random_starts=10)

def inp_to_dict(df,categoricals = {}):
    #divide df to df and y
    y = df["hr_OrderOfFinish"]
    df.drop("hr_OrderOfFinish",axis = 1,inplace = True)
    #y = y.apply(lambda x : 1 if x == 1 else 0)

    #split with train and test
    x1,x2,y1,y2 = train_test_split(df,y,test_size = 0.1)
    x2.to_csv("sample.csv")
    #x1,x2,y1,y2 = split_with_year(df,y,2017)
    del(df,y)
    gc.collect()

    x1_dic = {}
    x1_dic["main"] = x1.loc[:,~x1.columns.isin(categoricals)]

    print(x1.loc[:,~x1.columns.isin(categoricals)])
    for c in x1.columns:
        if c not in categoricals:
            continue
        x1_dic[c] = x1.loc[:,[c]]
    del(x1)
    gc.collect()

    x2_dic = {}
    x2_dic["main"] = x2.loc[:,~x2.columns.isin(categoricals)]
    print(x2.loc[:,x2.columns.isin(categoricals)])
    for c in x2.columns:
        if c not in categoricals:
            continue
        x2_dic[c] = x2.loc[:,[c]]
    del(x2)
    gc.collect()
    return x1_dic,x2_dic,y1,y2


def fillna(df,categoricals = [],verbose = False):
    df[df.isnull()] = np.nan
    df.dropna(subset = ["hr_OrderOfFinish"], inplace = True)

    if verbose:
        nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
        print(nan_rate)
    for f in df.columns:
        if f in categoricals:
            df[f].fillna("Nan",inplace = True)
        else:
            mean = df[f].mean()
            if np.isnan(mean):
                mean = 0
            df[f].fillna(mean,inplace = True)
    if verbose:
        nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
        print(nan_rate)
    return df

def normalize(df,numericals = []):
    remove = ["hr_OrderOfFinish","ri_Year"]
    nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
    print(nan_rate)

    for k in df.columns:
        if k in remove:
            continue
        if k in numericals:
            std = df[k].std()
            mean = df[k].mean()
            if np.isnan(std) or std == 0:
                print(k)
                df[k] = 0
            else:
                df[k] = (df[k] - mean)/std
    nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
    print(nan_rate)

    return df

def normalize_with_race(df,numricals = []):
    for k in numericals:
        df[k] = (df - df[k].mean())/df[k].std()
    return df

def label_encoding(df,categoricals = []):
    for k in categoricals:
        print(k)
        le = LabelEncoder()
        df[k] = le.fit_transform(df[k].astype(str))
    return df

def remove_unused(df,removes = []):
    columns = sorted(list(filter(lambda x : x not in removes, df.columns)))
    print(removes)
    return df[columns]

def preprocess_target(df):
    df.dropna(subset = ["hr_OrderOfFinish"],inplace = True)
    df = df[df["hr_OrderOfFinish"] != 0]
    df.loc[:,"hr_OrderOfFinish"] = df.loc[:,"hr_OrderOfFinish"].apply(lambda x : 1 if x == 1 else 0)
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

def load_dtypes(dtypes_path):
    dtypes = {}
    with open(dtypes_path) as fp:
        for row in fp.readlines():
            row_s = row.strip().split(",")
            name = row_s[0]
            dtyp = row_s[1]
            if name.startswith("hr_") and name != "hr_OrderOfFinish":
                continue
            dtypes[name] = dtyp
    return dtypes

def load_removes(remove_path):
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

def filter_df(df):
    remove = ["hi_RaceID","ri_Year"]
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
    del(con)
    gc.collect()
    #ret train_x,test_x,train_y,test_y
    return (train.loc[:,x.columns],test.loc[:,x.columns],train.loc[:,y.name],test.loc[:,y.name])

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

if __name__ == "__main__":
    main()
