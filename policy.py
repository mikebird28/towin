import preprep
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input, Embedding ,Concatenate, Flatten, SpatialDropout1D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from tqdm import tqdm
import logging
import gc
import time
import utils
import add_features
import preprocess
import argparse

REWARD_THREHOLD = 0.4


MODEL_NAME = "./models/policybased_model.h5"
PROCESS_FEATURE = utils.process_features() + ["hr_PaybackPlace_eval","hr_PaybackWin_eval"]
PROCESS_FEATURE = ["hi_Distance","ri_Year"] + PROCESS_FEATURE
PROCESS_FEATURE.remove("li_WinOdds")
DTYPE_PATH = "./data/dtypes.csv"

def main():
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(level = logging.DEBUG, filename = "cnn_exec.log", format = fmt)

    columns_dict = utils.load_dtypes(DTYPE_PATH)
    columns = sorted(columns_dict.keys())
    removes = utils.load_removes("./configs/remove.csv")
    categoricals,numericals = utils.cats_and_nums(columns,columns_dict)
    df = pd.read_csv("./data/output.csv",dtype = columns_dict,usecols = columns)
    df = df[df["ri_Year"] >= 2007]

    p = preprep.Preprep("./cache_files/fpolicy")
    p = p.add(preprocess_1, name = "preprocess1", cache_format = "feather")
    p = p.add(preprocess.fillna(),params = {"categoricals" : categoricals},name = "fillna",cache_format = "feather")
    p = p.add(label_encoding,params = {"categoricals" : categoricals},name = "lenc", cache_format = "feather")
    p = p.add(preprocess_2, name = "preprocess2", cache_format = "feather")
    p = p.add(preprocess.normalize(), params = {"categoricals" : categoricals}, name = "norm", cache_format = "feather")
    p = p.add(drop_columns, params = {"remove" : removes}, name = "drop", cache_format = "feather")
    df = p.fit_gene(df,verbose = True)
    df,test_df = split_with_year(df,year = 2017)
    del(p)
    gc.collect()
    policy_gradient(df,test_df,categoricals = categoricals)

def policy_gradient(df,test_df,categoricals = [], target_variable = "hr_PaybackPlace"):
    epoch_size = 100000
    batch_size = 512
    #batch_size = 256
    swap_interval = 1
    log_interval = 50
    lr_update_interval = 100000
    avg_loss = 0
    eval_variable = target_variable + "_eval"

    #feature informations
    all_features = [c for c in df.columns if c not in PROCESS_FEATURE]
    categoricals = [c for c in all_features if c in categoricals]
    numericals = [c for c in all_features if c not in categoricals]
    cats_max_values = categorical_max_values([df,test_df],categoricals)

    #prepare datasets
    df = remove_irregal(df,target = target_variable)
    test_df = remove_irregal(test_df,target = target_variable)

    bg = BatchGenerator(df,target_variable,categoricals)
    #train_batch = Batch(df,eval_variable, categoricals)
    #train_x,train_y = train_batch.get_all()
    test_batch = Batch(test_df,eval_variable,categoricals)
    test_x,test_y = test_batch.get_all()
    del(df,test_df,test_batch);gc.collect()

    #create model
    model,opt = nn(len(numericals),cats_max_values)
    tmp_model = keras.models.clone_model(model)

    x_holder = DataHolderX(batch_size,categoricals,numericals)
    y_holder = DataHolderY(batch_size,2)

    best_score = 0
    for i in range(1,epoch_size+1):
        rewards_elpased = time.time()
        races = bg.get_batch(batch_size,race_num = 2)

        for j in range(batch_size):
            x1,x2,y1,y2 = races[j].choose_one_others()
            x_holder.update(j, x1)
            y_holder.update(j, get_rewards_epsilon(tmp_model,x1,x2,y1,y2))
            #y_holder.update(j, get_rewards(tmp_model,x1,x2,y1,y2))

        rewards_elpased = time.time() - rewards_elpased
        fit_elapsed = time.time()
        history = model.fit(x_holder.get_dataset(), y_holder.get_dataset(), verbose = 0, epochs = 1,batch_size = batch_size)
        fit_elapsed = time.time() - fit_elapsed
        print("[{0}] Elapsed time : {1:.3f} (rewards : {2:.3f}, fit : {3:.3f})".format(i, rewards_elpased + fit_elapsed, rewards_elpased, fit_elapsed))
        avg_loss += history.history["loss"][0]

        if i % swap_interval == 0:
            tmp_model.set_weights(model.get_weights())

        if i % lr_update_interval == 0:
            current_lr = K.get_value(opt.lr)
            K.set_value(opt.lr,current_lr/2)

        if i % log_interval == 0:
            avg_loss = avg_loss/log_interval
            print(i)
            print("averate trian loss : {}".format(avg_loss))
            #evaluate(model,train_x,train_y)
            metrics = evaluate(model,test_x,test_y) #metrics = (hit_ratio, ret_ratio)
            if metrics[0] > best_score:
                best_score = metrics[0]
                model.save(MODEL_NAME)

            avg_loss = 0
            print()

class Batch(object):
    def __init__(self,df,target_variable,categoricals):
        all_features = [c for c in df.columns if c not in PROCESS_FEATURE]
        self.categoricals = sorted([c for c in all_features if c in categoricals])
        self.numericals = sorted([c for c in all_features if c not in categoricals])
        self.separate_idx = len(self.numericals)
        df = relocate_df(df,categoricals,self.numericals)
        self.x = remove_process_features(df).values
        self.y = df.loc[:,target_variable].values

    def get_all(self):
        x = convert_to_trainable(self.x,self.separate_idx, self.categoricals)
        return x,self.y

class BatchGenerator(object):
    def __init__(self,df,target_varibale,categoricals):
        all_features = [c for c in df.columns if c not in PROCESS_FEATURE]
        categoricals = sorted([c for c in all_features if c in categoricals])
        numericals = sorted([c for c in all_features if c not in categoricals])
        separate_idx = len(numericals)
        df = relocate_df(df,categoricals,numericals)

        self.races = []
        race_groups = df.groupby("hi_RaceID")
        self.races = np.ndarray(shape = (len(race_groups)), dtype = object)
        count = 0

        for idx,race in race_groups:
            x = remove_process_features(race)
            y = race.loc[:,target_varibale]
            race = Race(x.values,y.values,separate_idx,categoricals)
            self.races[count] = race
            count += 1

    def get_batch(self,batch_size,race_num = 1):
        if race_num == 1:
            races = np.random.choice(self.races,size = batch_size)
            return races
        else:
            size = (batch_size,race_num)
            races = np.random.choice(self.races,size = size).reshape([batch_size,race_num])
            races = concat_races(races)
            return races
        return races

class Race(object):
    def __init__(self,x,y,separate_idx,categoricals):
        self.x = x
        self.y = y
        self.separate_idx = separate_idx
        self.categoricals = categoricals

    def choose_one_others(self):
        target_idx = np.random.randint(0,len(self.x)) 
        idx = np.zeros(len(self.x), dtype = bool)
        idx[target_idx] = True

        x1 = convert_to_trainable(self.x[idx],self.separate_idx, self.categoricals)
        x2 = convert_to_trainable(self.x[~idx],self.separate_idx, self.categoricals)
        y1 = self.y[idx]
        y2 = self.y[~idx]
        return x1,x2,y1,y2

def concat_races(races):
    def f(rows):
        stacked_x = np.vstack([r.x for r in rows])
        stacked_y = np.hstack([r.y for r in rows])
        new_race = Race(stacked_x,stacked_y,rows[0].separate_idx,rows[0].categoricals)
        return new_race
    return np.apply_along_axis(f,1,races)

class DataHolderX(object):
    def __init__(self,batch_size,categoricals,numericals):
        self.dic = {}
        self.categoricals = categoricals 
        self.numericals = numericals
        for c in categoricals:
            self.dic[c] = np.zeros([batch_size,1])
        self.dic["main"] = np.zeros([batch_size,len(numericals)])

    def update(self,i,data_dict):
        self.dic["main"][i,:] = data_dict["main"]
        for c in self.categoricals:
            self.dic[c][i] = data_dict[c]

    def get_dataset(self):
        return self.dic

class DataHolderY(object):
    def __init__(self,batch_size,action_size):
        self.holder = np.zeros([batch_size,action_size])

    def update(self,i,value):
        self.holder[i] = value

    def get_dataset(self):
        return self.holder

def relocate_df(df,cats,nums):
    num_df = df.loc[:,nums]
    cat_df = df.loc[:,cats]
    others = df.loc[:,~df.columns.isin(cats + nums)]
    df = pd.concat([num_df,cat_df,others],axis = 1)
    return df

def remove_process_features(df):
    features = [c for c in df.columns if c not in PROCESS_FEATURE]
    return df.loc[:,features]

def convert_to_trainable(x,separate_idx,categoricals):
    x_dict = {}
    x_dict["main"] = x[:,0:separate_idx]
    for i in range(separate_idx,x.shape[1]):
        col = categoricals[i - separate_idx]
        x_dict[col] = x[:,i]
    return x_dict

def get_rewards_epsilon(model, x, x_dash, y, y_dash, epsilon = 0.05):
    horse_num = len(x_dash) + 1
    reward_matrix = np.ones([len(y_dash),2])
    reward_matrix[:,1] = y_dash

    #decide tickets to buy and get rewards
    epsilon_mask = np.random.binomial(1,epsilon,len(y_dash))
    pred = model.predict(x_dash)
    pred[:] = np.squeeze(pred)

    action = pred.argmax(axis = 1)
    action[:] = np.logical_xor(action,epsilon_mask).astype(np.int8)

    pred[:] = np.eye(2)[action] #convert pred to binary, reuse pred to avoid unnecessary memory allocaton
    pred *= reward_matrix #calculate each rewards

    #init reward 
    reward = np.ones(shape = [2])
    reward[1] = y

    reward += reward_matrix.sum()
    reward -= horse_num
    reward[:] += np.where(reward > REWARD_THREHOLD, 1 ,0)
    reward -= reward.mean()
    return reward

def get_rewards(model, x, x_dash, y, y_dash,iter_times = 30):
    reward_sum = np.zeros(shape = [2])
    action_reward = np.ones(shape = [2])
    action_reward[1] = y

    pred = model.predict(x_dash)
    pred[:] = np.squeeze(pred)

    horse_num = len(x_dash["main"])
    reward_matrix = np.ones([horse_num,2])
    reward_matrix[:,1] = y_dash

    horse_number = len(x_dash["main"]) + 1
    visited = np.ones_like(pred)

    puct_bias = np.zeros(shape = [horse_num,2])

    bin_pred = np.zeros(shape = [horse_num,2])
    buy_result = np.zeros(shape = [horse_num,2])
    reward = np.zeros(shape = [2])

    for i in range(iter_times):
        #calculate puct value (separate to two formula to avoid memory allcation
        #puct = win_num /visted + const * pred * np.sqrt(visited.sum(axis = 1, keepdims = True) - visited)
        puct_bias[:] = np.sqrt(visited.sum(axis = 1,keepdims = True) - visited)
        puct_bias /= (0.01 + visited)
        puct_bias *= pred

        #in case use puct value
        bin_pred[:] = get_action_by_puct(puct_bias)
        visited += bin_pred

        #caluculate reawrds (separate to multiple formula to avoid memory allocation)
        #buy_result[:] = bin_pred * reward_matrix
        buy_result[:] = bin_pred
        buy_result *= reward_matrix

        reward[:] = action_reward
        reward += buy_result.sum()
        reward -= horse_number
        reward_sum += np.where(reward > REWARD_THREHOLD, 1 ,0)

    reward_mean = reward_sum/iter_times
    reward_mean -= reward_mean.mean()
    return reward_mean 

def get_action_by_prob(pred,s,r,k):
    #rescale to consitent sum to 1
    s = pred.cumsum(axis=1)
    r = np.random.rand(pred.shape[0])
    k = (s.T < r).sum(axis=0).clip(0,1)
    #ret = np.zeros_like(pred)
    #ret[np.arange(pred.shape[0]),k] = 1

    #reuse s to avoid newly memory allocation
    s[:] = 0
    s[np.arange(pred.shape[0]),k] = 1
    return s

_eye_array = np.eye(2)
def get_action_by_puct(pred):
    return _eye_array[pred.argmax(axis = 1)]

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
    return (hit_ratio,ret_ratio)

def categorical_max_values(df_ls,categoricals):
    ret_dic = {}
    for df in df_ls:
        for c in df.columns:
            if c not in categoricals:
                continue
            max_value = df.loc[:,c].max()
            #ret_dic already had value and largaer than max_value, skip
            if c in ret_dic and ret_dic[c] > max_value:
                continue
            ret_dic[c] = max_value
    return ret_dic

# NN functions
def nn(numerical_features, max_values, learning_rate = 1e-4, l2_coef = 1e-4, unit_size = 1024, dropout_rate = 0.4):
    inputs = []
    flatten_layers = []

    #input unit
    #numericals:
    x = Input(shape = (numerical_features,) ,dtype = "float32", name = "main")
    inputs.append(x)
    flatten_layers.append(x)

    #categoricals
    for c in max_values:
        inp_dim = max_values[c] + 1
        out_dim = embedding_size(inp_dim)
        x = Input(shape = (1,),dtype = "float32",name = c)
        inputs.append(x)
        x = Embedding(inp_dim,out_dim,input_length = 1, embeddings_regularizer = keras.regularizers.l2(l2_coef))(x)
        x = SpatialDropout1D(0.2)(x)
        x = Flatten()(x)
        flatten_layers.append(x)
    x = Concatenate()(flatten_layers)

    x = Dense(units = unit_size, kernel_regularizer = regularizers.l2(l2_coef))(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(units = unit_size, kernel_regularizer = regularizers.l2(l2_coef))(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)

    x = BatchNormalization()(x)
    x = Dense(units = 2,kernel_regularizer = regularizers.l2(l2_coef), bias_regularizer = regularizers.l2(l2_coef))(x)
    #x = Dense(units = 2,kernel_regularizer = regularizers.l2(1e-2), bias_regularizer = regularizers.l2(1e-2))(x)
    x = Activation("softmax")(x)

    model = Model(inputs = inputs,outputs = x)
    opt = keras.optimizers.RMSprop(lr=learning_rate, rho = 0.9)
    model.compile(loss = log_loss, optimizer=opt)
    return model,opt

def log_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred,1e-8,1)
    return -tf.reduce_mean(tf.multiply(y_true,tf.log(y_pred)))

def embedding_size(size):
    if size < 10:
        return size // 2 + 1
    elif size < 30:
        return size // 5 + 1
    elif size < 150:
        return size // 10 + 1
    elif size < 300:
        return size // 15 + 1
    else:
        return size // 50 + 1

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

def drop_columns(df,remove = None, target = None,logger = None):
    logger = logger or logging.getLogger(__name__)
    logger.debug("drop_columns : function called")
    remove.append("has_li")
    remove.append("has_ti")
    remove.append("has_ri")
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

def preprocess_1(df):
    df = add_features.add_null_count(df)
    df = add_features.add_has_info(df)
    df = add_features.fix_extra_info(df)
    df = add_features.add_run_style_info(df)
    df = add_features.add_datetime_info(df)
    df = add_features.add_corner_info(df)
    df = add_features.add_delta_info(df)
    return df

def preprocess_2(df):
    df = add_features.add_season_info(df)
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


def remove_irregal(df,target = "hr_PaybackWin"):
    group_sum = df.loc[:,["hi_RaceID",target]].groupby("hi_RaceID")[target].sum().reset_index()
    irregals = group_sum.loc[group_sum.loc[:,target] == 0,"hi_RaceID"].values
    df = df.loc[~df.loc[:,"hi_RaceID"].isin(irregals),:]
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--learning-rate', action="store_const")
    main()
