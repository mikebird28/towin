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
from keras import regularizers
import gc
from keras import backend as K
import logging
import tensorflow as tf
#from utils import cats_and_nums, load_removes, load_dtypes

PROCESS_FEATUES = ["hi_RaceID","ri_Year","hr_OrderOfFinish"]
RACE_FEATURES = ["hi_Distance"]

def main():
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(level = logging.DEBUG, filename = "cnn_exec.log", format = fmt)

    columns = ["hi_RaceID","li_WinOdds","pred","hr_PaybackWin","hr_PaybackPlace"]
    df = pd.read_csv("./data/train_pred.csv",usecols = columns)
    test = pd.read_csv("./data/test_pred.csv",usecols = columns)
    #del(train,test);gc.collect()

    p = preprep.Preprep("./cache_files/rf")
    p = p.add(fillna(),name = "fillna")
    p = p.add(normalize(), name = "norm", cache_format = "feather")
    df = p.fit_gene(df,verbose = True)
    test_df = p.gene(test)
    
    target_variabe = "hr_PaybackWin"

    batch_size = 20000
    model = nn()
    tmp_model = model
    race_id = pd.Series(df["hi_RaceID"].unique())
    for i in range(batch_size):
        target_ids = race_id.sample(n = 128).values

        x_df = []
        y_df = []
        for idx in target_ids:
            race = df.loc[df.loc[:,"hi_RaceID"] == idx,:]
            train_record = race.sample(n = 1)
            x_df.append(to_trainable(train_record))
            train_idx = train_record.index[0]
            others = race.loc[race.index != train_idx,:]
            rewards = get_rewards(tmp_model,others,target = target_variabe)
            #y = pd.Series([1,train_record.loc[:,target_variabe].values[0]])
            y = pd.Series([1 + rewards, train_record.loc[:,target_variabe].values[0] + rewards])
            y = (y - len(race)).clip(0,None)
            y_df.append(y)

        
        x_df = pd.concat(x_df).values
        y_df = pd.concat(y_df,axis = 1).T.values
        model.fit(x_df,y_df)

        rewards = get_rewards(model,test_df)
        print(i)
        if i % 5 == 0:
            tmp_model = keras.models.clone_model(model)
            tmp_model.set_weights(model.get_weights())
            #print#(model.predict(to_trainable(test_df)))
            evaluate(model,df)
            evaluate(model,test_df)

def to_trainable(df):
    x = df.loc[:,["li_WinOdds","pred"]]
    return x

def get_rewards(model,df,target = "hr_PaybackWin"):
    pred = model.predict(to_trainable(df))
    row_maxes = pred.max(axis=1).reshape(-1, 1)
    bin_pred = np.where(pred == row_maxes, 1, 0)

    reward_matrix = pd.DataFrame(np.ones([len(pred),2]))
    reward_matrix.iloc[:,1] = df.loc[:,target].values
    rewards = (reward_matrix * bin_pred).sum().sum()
    return rewards


def evaluate(model,df,eval_key = "hr_PaybackWin_eval"):
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
    race_total = bin_pred.sum()
    print("buy")
    print(race_total)
    print("ret")
    print(ret_total/race_total)
    print("hit")
    print(hit_total/race_total)

def nn():
    inputs = Input(shape = (2,),dtype = "float32",name = "input")
    x = Dense(units = 6, kernel_regularizer = regularizers.l2(0.03))(inputs)
    x = Activation("relu")(x)
    #x = Dense(units = 8)(inputs)
    #x = Activation("relu")(x)
    x = Dense(units = 4,kernel_regularizer = regularizers.l2(0.03))(x)
    x = Activation("relu")(x)

    x = Dense(units = 2,kernel_regularizer = regularizers.l2(0.03))(x)
    x = Activation("softmax")(x)
    model = Model(inputs = inputs,outputs = x)
    opt = keras.optimizers.Adam(lr=0.001,epsilon = 1e-4)
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
                df.loc[:,f].fillna("Nan",inplace = True)
            elif mode == "fit":
                mean = df[f].mean()
                self.mean_dict[f] = mean
                if np.isnan(mean):
                    mean = 0
            else:
                mean = self.mean_dict[f]
            df.loc[:,f].fillna(mean,inplace = True)

        nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
        logger.debug("nan rate check at end of fillna : {}".format(nan_rate))
        return df

class normalize(preprep.Operator):
    def __init__(self):
        self.mean_dict = {}
        self.std_dict = {}

    def on_fit(self,df,numericals = [] ,logger = None):
        return self.normalize(df,"fit",numericals,logger)

    def on_pred(self,df,numericals = [] ,logger = None):
        return self.normalize(df,"pred",numericals,logger)

    def normalize(self,df,mode,numericals = [],logger = None):
        logger = logger or logging.getLogger(__name__)
        logger.debug("normalize : function called")

        remove = ["hr_OrderOfFinish","ri_Year","hi_RaceID","hr_PaybackWin","hr_PaybackPlace","hr_PaybackWin_eval","hr_PaybackPlace_eval"]
        nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
        logger.debug("nan rate befort normalization : {}".format(nan_rate))
        #targets = [c for c in df.columns if c not in remove]

        for k in df.columns:
            if k in remove:
                continue
            if k not in remove:
                if mode == "fit":
                    std = df[k].std()
                    mean = df[k].mean()
                    self.mean_dict[k] = mean
                    self.std_dict[k] = std
                else:
                    mean = self.mean_dict[k]
                    std = self.std_dict[k]
                if np.isnan(std) or std == 0:
                    logger.debug("std of {} is 0".format(k))
                    df.loc[:,k] = 0
                else:
                    df.loc[:,k] = (df.loc[:,k] - mean)/std
        nan_rate = 1 - df.isnull().sum().sum()/float(df.size)
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
