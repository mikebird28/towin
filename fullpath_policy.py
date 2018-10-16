import preprep
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import keras
from keras.models import Sequential,Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense,Activation,Dropout,Input,Conv2D,Concatenate,SpatialDropout1D,LSTM,Multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten,Lambda,Reshape
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.utils import np_utils
from keras import optimizers
import gc
from keras import backend as K
import logging
import tensorflow as tf
import random
import utils
#from utils import cats_and_nums, load_removes, load_dtypes

#PROCESS_FEATUES = ["hi_RaceID","ri_Year","hr_OrderOfFinish"]
PROCESS_FEATURE = utils.process_features() + ["hr_PaybackPlace_eval","hr_PaybackWin_eval"]
PROCESS_FEATURE = ["hi_Distance","ri_Year"] + PROCESS_FEATURE
DTYPE_PATH = "./data/dtypes.csv"

def main():
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(level = logging.DEBUG, filename = "cnn_exec.log", format = fmt)

    columns_dict = utils.load_dtypes(DTYPE_PATH)
    columns = sorted(columns_dict.keys())
    removes = utils.load_removes("remove.csv")
    categoricals,numericals = utils.cats_and_nums(columns,columns_dict)
    df = pd.read_csv("./data/output.csv",dtype = columns_dict,usecols = columns)

    p = preprep.Preprep("./cache_files/fpolicy")
    p = p.add(fillna(),params = {"categoricals" : categoricals},name = "fillna",cache_format = "feather")
    p = p.add(label_encoding,params = {"categoricals" : categoricals},name = "lenc", cache_format = "feather")
    p = p.add(normalize(), params = {"categoricals" : categoricals}, name = "norm", cache_format = "feather")
    p = p.add(drop_columns, params = {"remove" : removes}, name = "drop", cache_format = "feather")
    p = p.add(onehot_encoding,params = {"categoricals" : categoricals}, name = "ohe",cache_format = "feather")
    df = p.fit_gene(df,verbose = True)
    df,test_df = split_with_year(df,year = 2013)
    
    target_variabe = "hr_PaybackPlace"

    t = to_trainable(df)
    feature_size = t.shape[0]
    del(t);gc.collect();

    epoch_size = 100000
    batch_size = 128
    swap_interval = 50
    log_interval = 50

    model = nn()
    fit_func = fit(model)
    tmp_model = model
    race_id = pd.Series(df["hi_RaceID"].unique())

    for i in range(epoch_size):
        target_ids = race_id.sample(n = batch_size).values

        feature_size = 201
        x_df = np.zeors(shape = [batch_size,feature_size])
        y_df = np.zeros(shape = [batch_size,2])
        actions_df = np.zeros(shape = [batch_size])

        for j,idx in enumerate(target_ids):
            race = df.loc[df.loc[:,"hi_RaceID"] == idx,:]
            train_record = race.sample(n = 1)
            train_idx = train_record.index[0]

            trainable = to_trainable(train_record)
            x_df[j,:] = trainable

            others = race.loc[race.index != train_idx,:]
            rewards = get_rewards(tmp_model,others,target = target_variabe)
            hit_reward = train_record.loc[:,target_variabe].values[0] + rewards

            y_df[j,:] = [1+rewards, hit_reward]
            y_df[j,:] = y_df[j,:] - len(race)

            pred = np.squeeze(model.predict(trainable))
            actions_df[j] =  np.random.choice(2,p = pred)
        
        x_df = pd.concat(x_df).values
        actions_df = np_utils.to_categorical(actions_df, num_classes=2)
        y_df = np.where(y_df > 0.1,1,0)
        y_df = (actions_df * y_df).sum(axis = 1)

        fit_func([x_df,actions_df,y_df])

        if i % swap_interval == 0:
            tmp_model = keras.models.clone_model(model)
            tmp_model.set_weights(model.get_weights())

        if i % log_interval == 0:
            print(i)
            evaluate(model,df)
            evaluate(model,test_df)
            print()

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

def get_rewards(model,df,target = "hr_PaybackWin",iter_times = 30):
    M = 10
    reward_sum = 0
    pred = model.predict(to_trainable(df))
    pred = np.squeeze(pred)

    reward_matrix = pd.DataFrame(np.ones([len(df),2]))
    reward_matrix.iloc[:,1] = df.loc[:,target].values

    for i in range(M):
        bin_pred = get_action(model,df,threhold = 0.1,pred = pred)
        reward_sum += (reward_matrix * bin_pred).sum().sum()

    rewards = reward_sum/M
    return rewards 


def evaluate(model,df,eval_key = "hr_PaybackPlace_eval"):
    pred = model.predict(to_trainable(df))
    row_maxes = pred.max(axis=1).reshape(-1, 1)
    bin_pred = pd.DataFrame(np.where(pred == row_maxes, 1, 0))
    bin_pred = bin_pred.iloc[:,1]

    payoff_matrix = df.loc[:,eval_key]
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
    inputs = Input(shape = (201,),dtype = "float32",name = "input")
    l2_coef = 0.002

    x = Dense(units = 756, kernel_regularizer = regularizers.l2(l2_coef))(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(units = 756, kernel_regularizer = regularizers.l2(l2_coef))(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    #x = Dense(units = 4,kernel_regularizer = regularizers.l2(l2_coef))(x)
    #x = Activation("relu")(x)

    x = Dense(units = 2,kernel_regularizer = regularizers.l2(l2_coef))(x)
    x = Activation("softmax")(x)

    model = Model(inputs = inputs,outputs = x)
    opt = keras.optimizers.Adam(lr=0.01,epsilon = 1e-4)
    model.compile(loss = log_loss, optimizer=opt, metrics = ["mse"])
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
        df.loc[:,"hr_PaybackWin"] = (df.loc[:,"hr_PaybackWin"].fillna(0)/100).clip(0.0,50.0)
        df.loc[:,"hr_PaybackPlace"] = (df.loc[:,"hr_PaybackPlace"].fillna(0)/100).clip(0.0,50.0)

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

        remove = ["hr_OrderOfFinish","ri_Year","hi_RaceID","hr_PaybackWin","hr_PaybackPlace","hr_PaybackWin_eval","hr_PaybackPlace_eval"]
        numericals = [c for c in df.columns if c not in set(categoricals + remove)]

        nan_rate = 1 - df.loc[:,numericals].isnull().sum().sum()/float(df.size)
        logger.debug("nan rate befort normalization : {}".format(nan_rate))

        for k in numericals:
            if mode == "fit":
                mean = df[k].mean()
                std = df[k].std()
                self.mean_dict[k] = mean
                self.std_dict[k] = std
            else:
                mean = self.mean_dict[k]
                std = self.std_dict[k]

            if np.isnan(mean) or np.isnan(std) or std == 0:
                logger.debug("std of {} is 0".format(k))
                df.loc[:,k] = 0
            else:
                df.loc[:,k] = (df.loc[:,k] - mean)/std

        nan_rate = 1 - df.loc[:,numericals].isnull().sum().sum()/float(df.size)
        logger.debug("nan rate after normalization : {}".format(nan_rate))
        return df

def normalize_with_race(df,numericals = []):
    key = "hi_RaceID"
    groups = df.groupby(key)
    for i,group in groups:
        pass
    for k in numericals:
        df[k] = (df - df[k].mean())/df[k].std()
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


def _is_win(df):
    is_hit = df["pred"].idxmax() == df["hr_OrderOfFinish"].idxmax()
    return is_hit

def split_with_year(df,year):
    train = df[df["ri_Year"] < year]
    test = df[df["ri_Year"] >= year]
    gc.collect()
    #ret train_x,test_x,train_y,test_y
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

def batch_generator(df,batch_size,n):
    x = ["li_WinOdds","pred","is_padded"]
    y = ["hr_PaybackWin"]
    df = df.loc[:,x + y]

    indexes = np.unique(df.index.get_level_values(0).values)
    np.random.shuffle(indexes)
    #print(indexes.to_array())

    index_size = len(indexes)
    start_index = 0
    while True:
        reset_flag = False
        end_index = start_index + batch_size
        if end_index > index_size:
            end_index = index_size
            reset_flag = True

        target_indexes = indexes[start_index:end_index]
        batch_x = to_matrix(df.loc[target_indexes,x])
        batch_y = to_matrix(df.loc[target_indexes,y],for_y = True)

        start_index += batch_size
        if reset_flag:
            start_index = 0
            np.random.shuffle(indexes)
        yield batch_x,batch_y

def to_matrix(df,for_y = False):
    if for_y:
        dim1 = len(df.index.get_level_values(0).unique())
        dim2 = len(df.index.get_level_values(1).unique())
        result = df.values.reshape((dim1, dim2))
    else:
        dim1 = len(df.index.get_level_values(0).unique())
        dim2 = len(df.index.get_level_values(1).unique())
        result = df.values.reshape((dim1, dim2, df.shape[1]))
    return result

def playout(x,y,trial = 1000):
    avg = 0
    for i in range(trial):
        binary_array = np.zeros_like(y)
        for j in range(binary_array.shape[0]):
            random_select = np.random.multinomial(1,[1/18] * 18)
            binary_array[j] = random_select
        ret = np.sum(np.multiply(y,binary_array))
        avg += ret
    avg = avg/trial
    return avg

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true,y_pred)

def log_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred,1e-8,1)
    return -tf.reduce_mean(tf.multiply(y_true,tf.log(y_pred)))

if __name__ == "__main__":
    main()