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
    #targets = utils.load_targets("target.csv")
    years = range(2011,2018+1)
    df = pd.read_csv("./data/output.csv",dtype = columns_dict,usecols = columns)

    p = preprep.Preprep("./cache_files/cnn")
    p = p.add(preprocess_target, name = "target_pre")
    p = p.add(fillna,params = {"categoricals":categoricals},name = "fillna")
    p = p.add(label_encoding,params = {"categoricals":categoricals}, name = "lenc", cache_format = "feather")
    p = p.add(choose_columns,params = {"removes" : removes,"process_features" : PROCESS_FEATURES}, name = "remove", cache_format = "feather")
    p = p.add(choose_year,params = {"years" : years}, name = "years", cache_format = "feather")
    p = p.add(feature_engineering, params = {"categoricals" : categoricals}, name = "feng", cache_format = "feather")
    p = p.add(normalize, params = {"process_features" : PROCESS_FEATURES, "categoricals" : categoricals}, name = "norm", cache_format = "feather")
    p = p.add(inp_to_race,params = {"categoricals" : categoricals, "num_per_race" : 5}, name = "to_races", cache_format = "feather")
    p = p.add(inp_to_dict,params = {"categoricals" : categoricals, "target" : "margin"}, name = "to_dict", cache_format = "pickle")
    x1,x2,y1,y2 = p.fit_gene(df,verbose = True)
    del(p,df);gc.collect()

    x1 = filter_df(x1)
    x2 = filter_df(x2)
    nn(x1,x2,y1,y2)

def nn(x1,x2,y1,y2):
    y1 = np.log(y1 + 1)
    y2 = np.log(y2 + 1)
    print(x1.shape)
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
            #flatten_layers.append(x)
            flatten_layers = x
        else:
            dim_1 = x1[f].shape[1]
            x = Input(shape = (dim_1,),dtype = "float32",name = f)
            inputs.append(x)
            inp_dim = max_dict[f] + 1
            out_dim = int(emb_dim(inp_dim))
            x = Embedding(inp_dim,out_dim,input_length = dim_1, embeddings_regularizer = keras.regularizers.l2(0.01))(x)
            x = SpatialDropout1D(0.2)(x)
            #flatten_layers.append(x)

    #x = Concatenate()(flatten_layers)
    x = flatten_layers
    feature_size = x.shape[2].value
    x = Reshape([18,feature_size,1])(x)

    layer_num = 3
    layer_size = 512
    bn_axis = -1
    momentum = 0.0
    dropout_rate = 0.2
    activation = "relu"

    x = Conv2D(layer_size,[1,feature_size],padding = "valid", kernel_regularizer= keras.regularizers.l2(0.01))(x)
    x = Activation(activation)(x)
    x = BatchNormalization(axis = bn_axis,momentum = momentum)(x)
    x = Dropout(dropout_rate)(x)

    for i in range(layer_num):
        x = Conv2D(layer_size,[1,1],padding = "valid",kernel_regularizer= keras.regularizers.l2(0.01))(x)
        x = Activation(activation)(x)
        x = BatchNormalization(axis = bn_axis,momentum = momentum)(x)
        x = Dropout(dropout_rate)(x)

    x = Conv2D(1,[1,1],padding = "valid",kernel_regularizer= keras.regularizers.l2(0.01))(x)
    #x = Conv2D(layer_size,[1,1],padding = "valid")(x)
    x = Flatten()(x)

    #x = Dense(units = 1024)(x)
    #x = Activation(activation)(x)
    #x = Dense(units = 18)(x)
    #x = Activation("softmax")(x)

    model = Model(inputs = inputs,outputs = x)

    early_stopping = EarlyStopping(monitor='val_acc', patience= 50, verbose=0)
    check_point = ModelCheckpoint("./models/pair_nn.hd", period = 1, verbose = 0, save_best_only = True)

    #model.compile(loss = rank_sigmoid,optimizer="adam", metrics = [rank_accuracy])
    model.compile(loss='mse', optimizer='adam', metrics=[rank_accuracy])

    #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x1,y1,epochs = 50,batch_size = 32,validation_data = (x2,y2),callbacks = [early_stopping,check_point])
    print(model.predict(x2)[0])
    return model

def rank_sigmoid(y_true,y):
    dim0 = K.shape(y)[0]
    loss = tf.fill(tf.stack([dim0]), 0.0)

    label_gain = 10
    label_gain = 1
    pred_gain = 5.0
    pred_gain = 1.0
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
    bool_y_true = K.cast(K.equal(y_true,0),"int32")
    #bool_y_true = K.cast(K.equal(y_true,1),"int32")
    y_pred = 1 - K.sigmoid(y)
    return keras.metrics.categorical_accuracy(bool_y_true, y_pred)

def margin_accuracy(y_true,y_pred):
    y_bin = K.cast(K.equal(y_true,0.0),"int32")
    return keras.metrics.categorical_accuracy(y_bin, -y_pred)

def rank_loss(y_true,y):
    res = -y_true * K.log(y) + (1 - y_true) * K.log(1 - y)
    return res

def emb_dim(inp_dim):
    if inp_dim < 5:
        return inp_dim
    elif inp_dim < 20:
        return 10
    else:
        return 20

def inp_to_race(df,categoricals = [],num_per_race = 3,logger = None):
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

    categoricals = [c for c in df.columns if c in categoricals]
    numericals = [c for c in df.columns if c not in categoricals]
    #df.loc[:,numericals] = df.loc[:,numericals].clip(-10,10)
    #print(df.loc[:,numericals].head())

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
        df_new.loc[end:idx_count + max_group_size,"is_win"] = 0
        df_new.loc[end:idx_count + max_group_size,"is_place"] = 0
        df_new.loc[end:idx_count + max_group_size,"is_padded"] = 1

        df_new.loc[start:idx_count + max_group_size] = df_new.loc[start:idx_count + max_group_size].sample(frac = 1.0).values
        #print(df_new.loc[start:idx_count + max_group_size].head(18))
        idx_count += max_group_size 

    del(df,groups);
    gc.collect()
    df_new.index = pd.MultiIndex.from_product([range(total),range(max_group_size)])
    return df_new

def inp_to_dict(df, categoricals = {}, target = "is_win",logger = None):
    logger = logger or logging.getLogger(__name__)
    logger.debug("inp_to_dict : function_called")

    y = df.loc[:,[target]]
    df.drop(target,inplace = True, axis = 1)
    #x1,x2,y1,y2 = train_test_split(df,y,test_size = 0.1)
    x1,x2,y1,y2 = split_with_year(df,y,2017)
    del(df,y)
    gc.collect()

    x1_dic = {}

    remove = extract_columns(x1,PROCESS_FEATURES)
    x1.drop(remove,inplace = True, axis = 1)
    x2.drop(remove,inplace = True, axis = 1)
    numericals = [c for c in x1.columns if c not in categoricals]

    x1_dic["main"] = to_matrix(x1.loc[:,numericals])
    for c in x1.columns:
        if c not in categoricals:
            continue
        mtx = to_matrix(x1.loc[:,[c]])
        mtx = mtx.reshape([mtx.shape[0],mtx.shape[1]])
        x1_dic[c] = mtx
    y1 = to_matrix(y1)
    y1 = y1.reshape([y1.shape[0],y1.shape[1]])
    y1 = pd.DataFrame(y1)
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
    del(x2)
    gc.collect()
    return x1_dic,x2_dic,y1,y2

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
    #print(len(df.index.get_level_values(0)))
    dim1 = len(df.index.get_level_values(0).unique())
    dim2 = len(df.index.get_level_values(1).unique())
    result = df.values.reshape((dim1, dim2, df.shape[1]))
    return result

def optimize_params(df,categoricals):
    df = df[df["ri_Year"] >= 2013]
    metrics = "auc"
    count = 0

    def __objective(values):
        nonlocal count
        count += 1
        n_folds = 5 
        num_boost_round = 20000
        early_stopping_rounds = values[6]

        #kf = KFold(n = train_x.shape[0],n_folds = n_folds,random_state = 32,shuffle = True)
        kf = RaceKFold(df,n_folds)
     
        scores = []
        for i,(train,test) in enumerate(kf):
            x1,y1 = filter_df(train)
            x2,y2 = filter_df(test)

            features = x1.columns
            cats = list(filter(lambda x: x in categoricals,features))

            dtrain = lgb.Dataset(x1, label=y1,categorical_feature = cats)
            dtest  = lgb.Dataset(x2, label=y2,categorical_feature = cats)
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
            print("[{}] score ... {}".format(count,score))

        avg_score = sum(scores)/len(scores)
        print("avg_score : {}".format(avg_score))
        print()
        return -avg_score

    space = [
        Integer(3, 5, name='max_depth'),
        Integer(10, 300, name='num_leaves'),
        Integer(50, 80, name='min_child_samples'), #1
        Real(0.01, 1.0, name='subsample'),
        Real(0.01, 1.0, name='colsample_bytree'),
        Real(0, 5.0,name='lambda_l1'),
        Integer(50, 200, name='early_stopping_rounds'), #1
    ]
    res_gp = gp_minimize(__objective, space, n_calls=700,n_random_starts=1,xi = 0.005)
    print(res_gp)
    plot_convergence(res_gp)

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
