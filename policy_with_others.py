import preprep
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import keras
from keras.models import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense,Activation,Dropout,Input,Conv2D,Concatenate,SpatialDropout1D,LSTM,Multiply ,Add
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import optimizers
import gc
from keras import backend as K
import logging
import tensorflow as tf
import random
import utils
from multiprocessing import Pool

#PROCESS_FEATUES = ["hi_RaceID","ri_Year","hr_OrderOfFinish"]
PROCESS_FEATURE = utils.process_features() + ["hr_PaybackPlace_eval","hr_PaybackWin_eval"]
PROCESS_FEATURE = ["hi_Distance","ri_Year"] + PROCESS_FEATURE
PROCESS_FEATURE.remove("li_WinOdds")
DTYPE_PATH = "./data/dtypes.csv"

def main():
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(level = logging.DEBUG, filename = "cnn_exec.log", format = fmt)

    columns_dict = utils.load_dtypes(DTYPE_PATH)
    columns = sorted(columns_dict.keys())
    removes = utils.load_removes("remove.csv")
    categoricals,numericals = utils.cats_and_nums(columns,columns_dict)
    df = pd.read_csv("./data/output.csv",dtype = columns_dict,usecols = columns)
    df = df[df["ri_Year"] >= 2007]

    p = preprep.Preprep("./cache_files/fpolicy")
    p = p.add(fillna(),params = {"categoricals" : categoricals},name = "fillna",cache_format = "feather")
    p = p.add(label_encoding,params = {"categoricals" : categoricals},name = "lenc", cache_format = "feather")
    p = p.add(normalize(), params = {"categoricals" : categoricals}, name = "norm", cache_format = "feather")
    p = p.add(drop_columns, params = {"remove" : removes}, name = "drop", cache_format = "feather")
    p = p.add(onehot_encoding,params = {"categoricals" : categoricals}, name = "ohe",cache_format = "feather")
    df = p.fit_gene(df,verbose = True)
    df,test_df = split_with_year(df,year = 2017)
    
    target_variable = "hr_PaybackPlace"
    eval_variable = target_variable + "_eval"
    df = remove_irregal(df,target = target_variable)

    #prepare datasets
    train_x = to_trainable(df)
    train_y = df.loc[:,eval_variable]
    test_x = to_trainable(test_df)
    test_y = test_df.loc[:,eval_variable]
    bg = BatchGenerator(df,target_variable)
    del(p,df,test_df);gc.collect()

    feature_size = train_x.shape[0]

    epoch_size = 100000
    batch_size = 128
    swap_interval = 1
    log_interval = 500

    model = nn()

    tmp_model = keras.models.clone_model(model)

    feature_size = 202
    x_df = np.zeros(shape = [batch_size,feature_size])
    y_df = np.zeros(shape = [batch_size,2])

    for i in range(1,epoch_size+1):
        race_x,race_y = bg.get_batch(batch_size)

        for j in range(batch_size):
            x = race_x[j]
            y = race_y[j]
            target_row = x.sample(n = 1)
            target_idx = target_row.index[0]
            x_df[j,:] = target_row
            x_dash = x.loc[x.index != target_idx,:]
            y1 = y.loc[y.index == target_idx]
            y2 = y.loc[y.index != target_idx]
            y_df[j,:] = get_rewards(tmp_model,target_row,x_dash,y1,y2)

        model.fit(x_df,y_df,verbose = 0, epochs = 1,batch_size = batch_size)

        if i % swap_interval == 0:
            tmp_model.set_weights(model.get_weights())

        if i % log_interval == 0:
            print(i)
            evaluate(model,train_x,train_y)
            evaluate(model,test_x,test_y)
            print()

def get_rewards_wrapped(queue,i,model,target_row,x_dash,y1,y2):
    queue.put([i,get_rewards(model,target_row,x_dash,y1,y2)])

def fit(model):
    action_prob_placeholder = model.output
    action_onehot_placeholder = K.placeholder(shape=(None, 2), name="action")

    discount_reward_placeholder = K.placeholder(shape=(None,),name="reward")
    action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)

    log_action_prob = K.log(action_prob)
    loss = - log_action_prob * discount_reward_placeholder
    loss = K.mean(loss)

    adam = optimizers.Adam()
    updates = adam.get_updates(params=model.trainable_weights,loss=loss)
    train_fn = K.function(inputs=[model.input,action_onehot_placeholder,discount_reward_placeholder],outputs=[],updates=updates)
    return train_fn

def to_trainable(df):
    drop_targets = [c for c in df.columns if c in PROCESS_FEATURE]
    return df.drop(drop_targets,axis = 1)
    #x = df.loc[:,["li_WinOdds","pred"]]

class BatchGenerator(object):
    def __init__(self,df,target_varibale):
        self.x_df = []
        self.y_df = []

        for idx,race in df.groupby("hi_RaceID"):
            x = to_trainable(race)
            y = race.loc[:,target_varibale]
            self.x_df.append(x)
            self.y_df.append(y)

    def get_batch(self,batch_size):
        total_races = len(self.x_df)
        target_idx = random.sample(range(total_races),batch_size)
        x = [self.x_df[i] for i in target_idx]
        y = [self.y_df[i] for i in target_idx]
        return x,y

def pretrain(model,train_x,train_rew,test_x,test_rew):
    train_payoff = np.zeros(shape = [len(train_rew),2])
    train_payoff[:,1] = train_rew.values
    train_payoff[:,0] = (train_payoff[:,1] == 0).astype(int)
    #train_y = train_payoff
    train_y = train_payoff.clip(0,1)

    test_payoff = np.zeros(shape = [len(test_rew),2])
    test_payoff[:,1] = test_rew.values
    test_payoff[:,0] = (test_payoff[:,1] == 0).astype(int)
    #test_y = test_payoff.clip(0,1)

    epochs = 3
    for i in range(epochs):
        model.fit(train_x,train_y,verbose = 1, epochs = 1,batch_size = 128)
        evaluate(model,train_x,train_rew)
        evaluate(model,test_x,test_rew)

def get_action(model,df,threhold = 0.95,pred = None):
    #avoid duplicate caculation
    if pred is None:
        pred = model.predict(to_trainable(df))
        pred = np.squeeze(pred)

    actions = np.zeros_like(pred)
    for i in range(len(pred)):
        action_prob = random.random()
        if action_prob > threhold:
            #choose random action
            actions[i] = np.random.choice(2,p = pred[i])
        else:
            #choose greedy action
            row_max = pred[i].max()
            actions[i] = np.where(pred[i] == row_max,1,0)
    return actions

def get_rewards(model, x, x_dash, y, y_dash,iter_times = 10):
    reward_sum = np.zeros(shape = [2])
    action_reward = np.ones(shape = [2])
    action_reward[1] = y.values

    pred = model.predict(x_dash)
    pred = np.squeeze(pred)
    reward_matrix = np.ones([len(x_dash),2])
    reward_matrix[:,1] = y_dash.values

    for i in range(iter_times):
        horse_number = len(x_dash) + 1
        bin_pred = get_action(model,x_dash,threhold = 0.5,pred = pred)
        reward = action_reward + (reward_matrix * bin_pred).sum().sum() - horse_number
        reward = np.where(reward > 0.1, 1 ,0)
        reward_sum += reward

    reward_mean = reward_sum/iter_times
    #reward_mean = softmax(reward_mean)
    return reward_mean 

def softmax(z):
    s = np.max(z)
    e_x = np.exp(z - s)
    div = np.sum(e_x)
    return e_x / div

def evaluate(model,x,y):
    pred = model.predict(x)
    print(pred[0:10,:])
    row_maxes = pred.max(axis=1).reshape(-1, 1)

    bin_pred = np.where(pred == row_maxes, 1, 0)
    bin_pred = bin_pred[:,1]

    payoff_matrix = y
    hit_matrix = np.clip(payoff_matrix,0,1)

    ret = np.multiply(bin_pred,payoff_matrix)
    hit = np.multiply(bin_pred,hit_matrix)

    ret_total = np.sum(ret)
    hit_total = np.sum(hit)

    race_total = len(bin_pred)
    buy_total = bin_pred.sum()

    hit_ratio = hit_total/buy_total
    ret_ratio = ret_total/buy_total
    txt = "buy {}/{}, hit : {:.3f}, ret : {:.3f}".format(buy_total,race_total,hit_ratio,ret_ratio)
    print(txt)

def nn():
    inputs = Input(shape = (202,),dtype = "float32",name = "main")
    others = Input(shape = (18, 202,),dtype = "float32",name = "others")
    l2_coef = 2e-4
    UNIT_SIZE = 512

    x = inputs
    x = Dense(units = UNIT_SIZE, kernel_regularizer = regularizers.l2(l2_coef))(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Dense(units = 2,kernel_regularizer = regularizers.l2(l2_coef), bias_regularizer = regularizers.l2(l2_coef))(x)
    main_output = Activation("softmax")(x)

    #x = others


    model = Model(inputs = inputs,outputs = x)
    opt = keras.optimizers.RMSprop(lr=1e-4, rho = 0.99)
    model.compile(loss = log_loss, optimizer=opt)
    return model

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

        nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
        logger.debug("nan rate check at start of fillna : {}".format(nan_rate))

        for f in tqdm(df.columns):
            if f in categoricals:
                df.loc[:,f].fillna("nan",inplace = True)
            elif mode == "fit":
                mean = df[f].mean()
                if np.isnan(mean):
                    mean = 0
                self.mean_dict[f] = mean
                df.loc[:,f].fillna(mean,inplace = True)
            else:
                mean = self.mean_dict[f]
                df.loc[:,f].fillna(mean,inplace = True)
        nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
        logger.debug("nan rate check at end of fillna : {}".format(nan_rate))
        return df

def label_encoding(df,categoricals = [],logger = None):
    #init logger
    logger = logger or logging.getLogger(__name__)
    logger.debug("label_encoding : function called")

    dont_remove = ["hi_RaceID"]
    categoricals = [c for c in df.columns if c in categoricals]
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

def onehot_encoding(df,categoricals = [],logger = None):
    logger = logger or logging.getLogger(__name__)
    logger.debug("onehot_encoding : function called")

    remove_columns = ["ri_Datetime"] + PROCESS_FEATURE
    categoricals = [c for c in categoricals if c in df.columns]
    categoricals = [c for c in categoricals if c not in PROCESS_FEATURE]
    return df.drop(categoricals,axis = 1)

    for c in categoricals:
        if (c not in df.columns) or (c in remove_columns):
            continue
        logger.debug("onehot_encoding : start encoding {}".format(c))
        ohe = OneHotEncoder(sparse = False)
        transformed = pd.DataFrame(ohe.fit_transform(df.loc[:,[c]]))
        transformed.columns = ["{}_{}".format(c,i+1) for i in range(len(transformed.columns))]
        transformed_dims = transformed.shape[1]
        df = pd.concat([df,transformed],axis = 1)
        df.drop(c,axis = 1,inplace = True)
        df.loc[:,transformed.columns] = df.loc[:,transformed.columns].astype(np.int8)
        logger.debug("onehot_encoding : finish encoding {}, dim size = {}".format(c,transformed_dims))
    return df


def drop_columns(df,remove = None, target = None,logger = None):
    logger = logger or logging.getLogger(__name__)
    logger.debug("drop_columns : function called")
    if (remove is not None) and (target is not None):
        raise ValueError("you can use only 'remove' or 'target' either")
    elif remove is not None:
        logger.debug("drop_columns : remove columns")
        columns = [c for c in df.columns if c not in remove]
        return df.loc[:,columns]
    elif target is not None:
        logger.debug("drop_columns : extract target columns")
        return df.loc[:,target]
    else:
        logger.debug("drop_columns : do nothing")
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

        special_columns = ["li_WinOdds","ri_Distance"]
        remove = ["hr_OrderOfFinish","ri_Year","hi_RaceID","hr_PaybackWin","hr_PaybackPlace","hr_PaybackWin_eval","hr_PaybackPlace_eval"] + special_columns
        numericals = [c for c in df.columns if c not in set(categoricals + remove)]

        nan_rate = 1 - df.loc[:,numericals].isnull().sum().sum()/float(df.size)
        logger.debug("nan rate before normalization : {}".format(nan_rate))

        df = _norm_with_race(df,numericals)
        df = _norm_with_df(df,special_columns)
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

def _is_win(df):
    is_hit = df["pred"].idxmax() == df["hr_OrderOfFinish"].idxmax()
    return is_hit

def split_with_year(df,year):
    train = df[df["ri_Year"] < year]
    test = df[df["ri_Year"] >= year]
    return (train,test)

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

def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true,y_pred)

def log_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred,1e-8,1)
    return -tf.reduce_mean(tf.multiply(y_true,tf.log(y_pred)))

def remove_irregal(df,target = "hr_PaybackWin"):
    group_sum = df.loc[:,["hi_RaceID",target]].groupby("hi_RaceID")[target].sum().reset_index()
    irregals = group_sum.loc[group_sum.loc[:,target] == 0,"hi_RaceID"].values
    df = df.loc[~df.loc[:,"hi_RaceID"].isin(irregals),:]
    return df

if __name__ == "__main__":
    main()