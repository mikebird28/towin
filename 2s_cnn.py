import preprep
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential,Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense,Activation,Dropout,Input,Conv2D,Concatenate,SpatialDropout1D,LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten,Lambda,Reshape
from keras.layers.embeddings import Embedding
import gc
from keras import backend as K
import utils
import logging
import tensorflow as tf
#from utils import cats_and_nums, load_removes, load_dtypes

PROCESS_FEATUES = ["hi_RaceID","ri_Year","hr_OrderOfFinish"]
RACE_FEATURES = ["hi_Distance"]

def main():
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(level = logging.DEBUG, filename = "cnn_exec.log", format = fmt)

    columns_dict = utils.load_dtypes("dtypes.csv")
    columns = sorted(columns_dict.keys())
    
    categoricals,numericals = utils.cats_and_nums(columns,columns_dict)
    removes = utils.load_removes("remove.csv")
    #targets = utils.load_targets("target.csv")
    years = range(2010,2018+1)
    df = pd.read_csv("output.csv",dtype = columns_dict,usecols = columns)

    p = preprep.Preprep("./cache_2s")
    p = p.add(preprocess_target, name = "target_pre")
    p = p.add(fillna,params = {"categoricals":categoricals},name = "fillna")
    p = p.add(label_encoding,params = {"categoricals":categoricals}, name = "lenc", cache_format = "feather")
    p = p.add(choose_columns,params = {"removes" : removes}, name = "remove", cache_format = "feather")
    #p = p.add(choose_columns,params = {"targets" : targets}, name = "remove", cache_format = "feather")
    p = p.add(choose_year,params = {"years" : years}, name = "years", cache_format = "feather")
    p = p.add(normalize, params = {"numericals" : numericals}, name = "norm", cache_format = "feather")
    p = p.add(add_race_info, params = {"numericals" : numericals}, name = "add_race", cache_format = "feather")
    p = p.add(inp_to_race,params = {"numericals":numericals,"categoricals" : categoricals, "num_per_race" : 5}, name = "to_races", cache_format = "feather")
    p = p.add(inp_to_dict,params = {"categoricals" : categoricals}, name = "to_dict", cache_format = "pickle")
    x1,x2,y1,y2 = p.fit_gene(df,verbose = True)
    del(p,df);gc.collect()

    x1 = filter_df(x1)
    x2 = filter_df(x2)
    nn(x1,x2,y1,y2)

def nn(x1,x2,y1,y2):
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
        else:
            dim_1 = x1[f].shape[1]
            x = Input(shape = (dim_1,),dtype = "float32",name = f)
            inputs.append(x)
            inp_dim = max_dict[f] + 1
            out_dim = int(emb_dim(inp_dim))
            x = Embedding(inp_dim,out_dim,input_length = dim_1)(x)
            x = SpatialDropout1D(0.2)(x)
            #x = Flatten()(x)
        flatten_layers.append(x)
    x = Concatenate()(flatten_layers)
    feature_size = x.shape[2].value
    x = Reshape([18,feature_size,1])(x)

    layer_num = 1
    layer_size = 128
    bn_axis = -1
    momentum = 0.95
    dropout_rate = 0.4
    activation = "relu"

    x = Conv2D(layer_size,[1,feature_size],padding = "valid")(x)
    x = Activation(activation)(x)
    x = BatchNormalization(axis = bn_axis,momentum = momentum)(x)
    x = Dropout(dropout_rate)(x)

    for i in range(layer_num):
        x = Conv2D(layer_size,[1,1],padding = "valid")(x)
        x = Activation(activation)(x)
        x = BatchNormalization(axis = bn_axis,momentum = momentum)(x)
        x = Dropout(dropout_rate)(x)

    """
    x = Reshape([18,layer_size])(x)
    x = LSTM(layer_size,input_shape = (18,layer_size),return_sequences = True)(x)
    x = Flatten()(x)
    #x = Reshape([18,1,layer_size])(x)
    #x = BatchNormalization()(x)
    #x = Dropout(dropout_rate)(x)
    """

    
    x = Conv2D(1,[1,1],padding = "valid")(x)
    x = Flatten()(x)
    """
    x = Dense(units = 200)(x)
    x = Activation(activation)(x)
    x = Dropout(0.2)(x)
    """

    #x = Dense(units = 18)(x)

    #x = Activation("sigmoid")(x)

    model = Model(inputs = inputs,outputs = x)
    opt = keras.optimizers.Adam(lr=0.001,epsilon = 1e-4)
    early_stopping = EarlyStopping(monitor='val_rank_accuracy', patience= 50, verbose=0)
    check_point = ModelCheckpoint("./models/pair_nn.hd", period = 1, verbose = 0, save_best_only = True)
    #model.compile(loss = kokon_loss,optimizer=opt, metrics = [rank_accuracy])
    model.compile(loss = rank_sigmoid,optimizer=opt, metrics = [rank_accuracy])
    #model.compile(loss = "categorical_crossentropy",optimizer=opt, metrics = ["accuracy"])
    model.fit(x1,y1,epochs = 200,batch_size = 32,validation_data = (x2,y2),callbacks = [early_stopping,check_point])
    return model

def rank_sigmoid(y_true,y):
    dim0 = K.shape(y)[0]
    loss = tf.fill(tf.stack([dim0]), 0.0)

    label_gain = 10
    pred_gain = 5.0
    for i in range(18-1):
        for j in range(i+1,18):
            l = K.sigmoid(pred_gain * (y[:,i] - y[:,j]))
            l_true = K.sigmoid(label_gain * (y_true[:,i] - y_true[:,j]))
            entropy = -l_true * K.log(K.clip(l,1e-8,1.0)) - (1 - l_true) * K.log(K.clip(1 - l,1e-8,1.0)) 
            loss = loss + entropy  
    loss = K.reshape(loss,[-1,1])/18
    return loss

def kokon_loss(y_true,y):
    dim0 = K.shape(y)[0]
    loss = tf.fill(tf.stack([dim0]),0.0)
    for i in range(18):
        for j in range(18):
        #for j in range(i+1,18):
            softplus = K.softplus(y[:,j] - y[:,i])
            tanh = K.tanh(y_true[:,j] - y_true[:,i])
            loss = loss + K.clip(18 + softplus * tanh,0.0,18.0)
    loss = K.reshape(loss,[-1,1])/(18*18)
    return loss

def rank_accuracy(y_true,y):
    bool_y_true = K.cast(K.equal(y_true,1),"int32")
    y_pred = 1 - K.sigmoid(y)
    return keras.metrics.categorical_accuracy(bool_y_true, y_pred)

def rank_loss(y_true,y):
    res = -y_true * K.log(y) + (1 - y_true) * K.log(1 - y)
    return res

def emb_dim(inp_dim):
    if inp_dim < 10:
        return inp_dim
    elif inp_dim < 20:
        return 10
    else:
        return 20

def inp_to_race(df,numericals = [],categoricals = [],num_per_race = 3,logger = None):
    #init logger
    logger = logger or logging.getLogger(__name__)
    logger.debug("inp_to_race : function called")

    key = "hi_RaceID"
    groups = df.groupby(key)
    
    max_group_size = 18
    total = len(groups)
    size = len(groups) * max_group_size
    df["is_padded"] = 0

    columns = df.columns
    df_new = pd.DataFrame(np.zeros((size, len(columns))),columns = columns)

    idx_count = 0

    categoricals = extract_columns(df,categoricals)
    numericals = extract_columns(df,numericals)
    if "ri_Year" not in numericals:
        numericals.append("ri_Year")

    for name,group in tqdm(groups):
        start = idx_count
        end = idx_count + len(group)
        logger.debug("inp_to_race : grouped {}, original race size was {}".format(name,len(group)))

        df_new.iloc[start:end,:] = group.values
        mean = group[numericals].mean().values
        df_new.loc[end:idx_count + max_group_size,numericals] = mean
        df_new.loc[end:idx_count + max_group_size,"hr_OrderOfFinish"] = 19
        df_new.loc[end:idx_count + max_group_size,"is_padded"] = 1

        df_new.loc[start:idx_count + max_group_size] = df_new.loc[start:idx_count + max_group_size].sample(frac = 1.0).values
        #print(df_new.loc[start:idx_count + max_group_size].head(18))
        idx_count += max_group_size 

    del(df,groups);
    gc.collect()
    df_new.index = pd.MultiIndex.from_product([range(total),range(max_group_size)])
    return df_new

def inp_to_dict(df,categoricals = {},logger = None):
    logger = logger or logging.getLogger(__name__)
    logger.debug("inp_to_dict : function_called")

    target = "hr_OrderOfFinish"
    y = df.loc[:,[target]]
    df.drop([target],inplace = True,axis = 1)
    #split with train and test
    #x1,x2,y1,y2 = train_test_split(df,y,test_size = 0.1)
    x1,x2,y1,y2 = split_with_year(df,y,2017)
    del(df,y)
    gc.collect()

    x1_dic = {}

    remove = ["hr_OrderOfFinish","hi_RaceID","ri_Year","ri_RaceID",]
    remove = extract_columns(x1,remove)
    x1.drop(remove,inplace = True, axis = 1)
    x2.drop(remove,inplace = True, axis = 1)
    numericals = [c for c in x1.columns if c not in categoricals]
    print([c for c in x1.columns])

    x1_dic["main"] = to_matrix(x1.loc[:,numericals])
    for c in x1.columns:
        if c not in categoricals:
            continue
        mtx = to_matrix(x1.loc[:,[c]])
        mtx = mtx.reshape([mtx.shape[0],mtx.shape[1]])
        x1_dic[c] = mtx
    y1 = to_matrix(y1)
    y1 = y1.reshape([y1.shape[0],y1.shape[1]])
    #y1 = order_to_target(y1)
    del(x1)
    gc.collect()

    x2_dic = {}
    x2_dic["main"] = to_matrix(x2.loc[:,numericals])
    for c in x2.columns:
        if c not in categoricals:
            continue
        mtx = to_matrix(x2.loc[:,[c]])
        mtx = mtx.reshape([mtx.shape[0],mtx.shape[1]])
        x2_dic[c] = mtx
    y2 = to_matrix(y2)
    y2 = y2.reshape([y2.shape[0],y2.shape[1]])
    #y2 = order_to_target(y2)
    del(x2)
    gc.collect()
    return x1_dic,x2_dic,y1,y2

def fillna(df,categoricals = [], logger = None):
    logger = logger or logging.getLogger(__name__)
    logger.debug("fillna : function called")

    df[df.isnull()] = np.nan
    df.dropna(subset = ["hr_OrderOfFinish"], inplace = True)

    nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
    logger.debug("nan rate check at start of fillna : {}".format(nan_rate))

    for f in tqdm(df.columns):
        if f in categoricals:
            df.loc[:,f].fillna("Nan",inplace = True)
        else:
            mean = df[f].mean()
            if np.isnan(mean):
                mean = 0
            df.loc[:,f].fillna(mean,inplace = True)
    nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
    logger.debug("nan rate check at end of fillna : {}".format(nan_rate))
    return df

def normalize(df,numericals = [],logger = None):
    logger = logger or logging.getLogger(__name__)
    logger.debug("normalize : function called")

    remove = ["hr_OrderOfFinish","ri_Year","hi_RaceID"]
    nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
    logger.debug("nan rate befort normalization : {}".format(nan_rate))

    for k in df.columns:
        if k in remove:
            continue
        if k in numericals:
            std = df[k].std()
            mean = df[k].mean()
            if np.isnan(std) or std == 0:
                logger.debug("std of {} is 0".format(k))
                df.loc[:,k] = 0
            else:
                df.loc[:,k] = (df.loc[:,k] - mean)/std
    nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
    logger.debug("nan rate after normalization : {}".format(nan_rate))
    return df

def add_race_info(df,numericals):
    numericals = [c for c in numericals if c not in PROCESS_FEATUES]
    numericals = extract_columns(df,numericals)
    for c in numericals:
        columns_ls = ["hi_RaceID",c]
        mean = df.loc[:,columns_ls].groupby("hi_RaceID").mean()
        mean.columns = [c + "_mean"]
        df = df.merge(mean,on = "hi_RaceID",how = "left")
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

def choose_columns(df,removes = None,targets = None):
    if removes != None and targets != None:
        raise Exception("you can use only either of removes or taget")
    elif removes != None:
        columns = sorted(list(filter(lambda x : x not in removes, df.columns)))
        return df.loc[:,columns]
    elif targets != None:
        targets = sorted(list(set(targets + PROCESS_FEATUES)))
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

    #drop newbies
    before_size = len(df)
    df = df.loc[df.loc[:,"hi_CourseCode"] == "05s",:]
    df = df.loc[~df.loc[:,"hi_ConditionClass"].isin(["0s","s"]),:]
    after_size = len(df)
    logger.debug("preprocess_target : drop newbies, {} => {}".format(before_size,after_size))
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
    remove = ["hr_OrderOfFinish","hi_RaceID","ri_Year"]
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
    #print(df.shape)
    #print(len(df.index.get_level_values(0)))
    dim1 = len(df.index.get_level_values(0).unique())
    dim2 = len(df.index.get_level_values(1).unique())
    result = df.values.reshape((dim1, dim2, df.shape[1]))
    return result

def order_to_target(y):
    #equal_array = (np.equal(y[:,0],y[:,1]).astype(np.float32)) *  0.5
    #less_array = np.less(y[:,0],y[:,1]).astype(np.float32)
    return  np.equal(y,1).astype(np.float32)
    #return equal_array + less_array

if __name__ == "__main__":
    main()
