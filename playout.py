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

    columns = ["hi_RaceID","li_WinOdds","pred","hr_PaybackWin","hr_PaybackPlace"]
    df = pd.read_csv("./data/train_pred.csv",usecols = columns)
    test = pd.read_csv("./data/test_pred.csv",usecols = columns)
    #del(train,test);gc.collect()

    p = preprep.Preprep("./cache_files/playout")
    p = p.add(fillna(),name = "fillna")
    p = p.add(normalize(), name = "norm", cache_format = "feather")
    p = p.add(inp_to_race, name = "to_races", cache_format = "feather")
    df = p.fit_gene(df,verbose = True)
    test_df = p.gene(test)
    print(test_df.describe())
    del(p);gc.collect()


    target = "hr_PaybackWin"
    x1 = to_matrix(df.loc[:,["li_WinOdds","pred","is_padded"]])
    y1 = to_matrix(df.loc[:,[target]],for_y = True)
    y1e = to_matrix(df.loc[:,[target+"_eval"]],for_y = True)
    x2 = to_matrix(test_df.loc[:,["li_WinOdds","pred","is_padded"]])
    y2 = to_matrix(test_df.loc[:,[target]],for_y = True)
    y2e = to_matrix(test_df.loc[:,[target+"_eval"]],for_y = True)
 
    #y = to_matrix(df.loc[:,["hr_PaybackPlace"]],for_y = True)

    model = nn()
    model.fit(x1,y1,epochs = 30,batch_size = 32, validation_data = [x2,y2])
    print("inference finished")
    evaluate(x1,y1e,model)
    evaluate(x2,y2e,model)
    #gene = batch_generator(df,10000,100)

    """
    for x,y in gene:
        #x_main = x[0]
        #y_main = softmax(y[0] + playout(x,y))
        #print(y_main)

        model.fit(x,y,epochs = 1,batch_size = 32)
        #model.fit(x,y,epochs = 1,batch_size = 32,validation_data = (x2,y2))
    """

def evaluate(x,y,model):
    pred = model.predict(x)
    bin_pred = np.zeros_like(pred)
    bin_pred[np.arange(len(pred)),pred.argmax(axis = 1)] = 1

    payoff_matrix = y
    hit_matrix = np.clip(y,0,1)

    ret = np.multiply(bin_pred,payoff_matrix)
    hit = np.multiply(bin_pred,hit_matrix)

    ret_total = np.sum(ret)
    hit_total = np.sum(hit)
    race_total = len(y)
    print(ret_total/race_total)
    print(hit_total/race_total)

def nn():

    activation = "relu"
    dropout_rate = 0.3
    inputs = Input(shape = (18,3),dtype = "float32",name = "input")
    x = Reshape([18,3,1])(inputs)

    x = Conv2D(16,[1,3],padding = "valid")(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    for i in range(8):
        x = Conv2D(16,[1,1],padding = "valid")(x)
        x = Activation(activation)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    x = Conv2D(1,[1,1],padding = "valid")(x)
    x = Flatten()(x)

    model = Model(inputs = inputs,outputs = x)
    opt = keras.optimizers.Adam(lr=0.001,epsilon = 1e-4)
    model.compile(loss = "mse",optimizer=opt, metrics = ["mse"])
    #model.fit(x1,y1,epochs = 200,batch_size = 32,validation_data = (x2,y2))
    return model

def emb_dim(inp_dim):
    if inp_dim < 10:
        return inp_dim
    elif inp_dim < 20:
        return 10
    else:
        return 20

def inp_to_race(df,logger = None):
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

    for name,group in tqdm(groups):
        start = idx_count
        end = idx_count + len(group)
        logger.debug("inp_to_race : grouped {}, original race size was {}".format(name,len(group)))

        df_new.iloc[start:end,:] = group.values
        mean = group.mean().values
        df_new.loc[end:idx_count + max_group_size,:] = mean
        df_new.loc[end:idx_count + max_group_size,"hr_PaybackWin"] = 0
        df_new.loc[end:idx_count + max_group_size,"hr_PaybackPlace"] = 0
        df_new.loc[end:idx_count + max_group_size,"is_padded"] = 1

        df_new.loc[start:idx_count + max_group_size] = df_new.loc[start:idx_count + max_group_size].sample(frac = 1.0).values
        #print(df_new.loc[start:idx_count + max_group_size].head(18))
        idx_count += max_group_size 

    del(df,groups);
    gc.collect()
    df_new.index = pd.MultiIndex.from_product([range(total),range(max_group_size)])
    return df_new

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
        df.loc[:,"hr_PaybackWin"] = (df.loc[:,"hr_PaybackWin"].fillna(0)/1000).clip(0.0,1.0)
        df.loc[:,"hr_PaybackPlace"] = (df.loc[:,"hr_PaybackPlace"].fillna(0)/1000).clip(0.0,1.0)

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

"""
def evaluate(df,y,model,features):
    df = pd.concat([df,y],axis = 1)
    df["pred"] = model.predict(filter_df(df[features]))
    is_hit = df[["hi_RaceID","pred","hr_OrderOfFinish"]].groupby("hi_RaceID").apply(_is_win)
    score = sum(is_hit)/len(is_hit)
    print(score)
"""

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

if __name__ == "__main__":
    main()
