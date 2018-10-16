import lightgbm as lgb
import preprep
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,KFold
#from skopt.space import Real,Integer
#from skopt.plots import plot_convergence
#from skopt import gp_minimize
import numpy as np
import gc
import utils
from tqdm import tqdm
import logging

import add_features
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

PROCESS_FEATURE = utils.process_features()
TQDM_DISABLE = True

def main():
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(level = logging.INFO, filename = "lgb_exec.log", format = fmt)
    cache_mode = False

    mode = "normal"
    #mode = "tuning"

    #target_variable = "norm_time"
    #target_variable = "norm_margin"
    #target_variable = "div_time"
    #target_variable = "norm_time"
    target_variable = "margin"
    objective = "regression"

    columns_dict = utils.load_dtypes("./data/dtypes.csv")
    columns = sorted(columns_dict.keys())
    categoricals,numericals = utils.cats_and_nums(columns,columns_dict)
    categoricals = categoricals + add_features.additional_categoricals()
    removes = utils.load_removes("remove.csv")
    #removes = load_removes("remove_new.csv")

    df = pd.read_csv("./data/output.csv",dtype = columns_dict,usecols = columns)
    df = df.sort_index(axis=1, ascending=True)
    p = preprep.Preprep("./cache_files/lgb")
    p = p.add(preprocess,name = "preprocess",cache_format = "feather")
    p = p.add(utils.replace_few, params = {"categoricals" : categoricals} , cache_format = "feather", name ="replace_few")
    p = p.add(label_encoding,params = {"categoricals":categoricals},cache_format = "feather",name = "lenc")
    p = p.add(remove_unused,params = {"removes" : removes},cache_format = "feather",name = "remove")
    p = p.add(feature_engineering,params = {"numericals":numericals,"categoricals" :categoricals},cache_format = "feather",name = "feng")
    p = p.add(utils.remove_illegal_races, params = {}, cache_format = "feather", name = "rem_illegals")
    df = p.fit_gene(df,verbose = True)

    df.dropna(subset = ["margin"],inplace = True)
    df[target_variable] = np.log(1 + df[target_variable])

    if mode == "normal":
        kf = RaceKFold(df,n = 10)
        score = Score()
        for i,(df1,df2) in enumerate(kf):
            if i == 0:
                model = light_gbm(df1,df2,categoricals,show_importance = True,score = score,target_variable = target_variable, objective = objective)
                output_predicted_dataset(df1,df2,model)
            else:
                light_gbm(df1,df2,categoricals,show_importance = False,score = score,target_variable = target_variable, objective = objective)
                #light_gbm(df1,df2,categoricals,show_importance = True,score = score)

            del(df1,df2);gc.collect()
        score.show()
    else:
        optimize_params(df,categoricals,target_variable = target_variable,objective = objective,metrics = "l2")

def light_gbm(df1,df2,categoricals,score = None,show_importance = True,objective = "binary",target_variable = "is_win",logger = None):
    logger = logger or logging.getLogger(__name__)
    logger.info("start training")

    x1,y1 = filter_df(df1,train = True, target_variable = target_variable)
    x2,y2 = filter_df(df2,target_variable = target_variable)
    if objective == "binary":
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': objective,
            'metric': 'mse',
            'max_depth': 3,
            'num_leaves': 800,
            'min_child_samples': 80,
            'scale_pos_weight': 1.0,
            'subsample': 0.77,
            'colsample_bytree': 0.37,
            'lambda_l2' : 0.0,
            'lambda_l1' : 5.0,

            'num_iterations' : 20000,
            'max_delta_step' : 20,

            'bagging_freq': 1,
            'learning_rate': 0.01,
            'verbose': -1,
        }
    else:
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': objective,
            'metric': 'mse',
            'max_depth': 135,
            'num_leaves': 52,
            'min_child_samples': 85,
            'scale_pos_weight': 1.0,
            'subsample': 0.61,
            'colsample_bytree': 0.245,
            'lambda_l2' : 0.0,
            'lambda_l1' : 2.8,

            'num_iterations' : 20000,
            'max_delta_step' : 20,

            'bagging_freq': 1,
            'learning_rate': 0.01,
            'verbose': -1,
        }

    features = x1.columns
    categoricals = list(filter(lambda x: x in categoricals,features))
 
    dtrain = lgb.Dataset(x1[features], label=y1,categorical_feature = categoricals)
    dtest  = lgb.Dataset(x2[features], label=y2,categorical_feature = categoricals)
    #[5, 50, 1, 1.0, 0.8, 0.3]
    #[5, 100, 2, 1.0, 0.8,0.3]
    model = lgb.train(params,
            dtrain,
            valid_sets=[dtrain, dtest],
            valid_names=['train','valid'],
            evals_result= {},
            num_boost_round=20000,
            early_stopping_rounds= 300,
            verbose_eval=50,
    )
    #score = model.best_score["valid"]["auc"]
    if show_importance:
        importance = zip(x1.columns,model.feature_importance())
        importance = sorted(importance, key = lambda x : x[1])
        for k,v in importance:
            msg = "{} : {}".format(k,v)
            print(msg)
            logger.info(msg)
        #score_threhold(model,df2)

    win_hit,place_hit,win_return,place_return = evaluate(df2,model,features,show_results = True,mode = "argmin")
    if score is not None:
        score.register(win_hit,place_hit,win_return,place_return)
    return model

def optimize_params(df,categoricals,objective = "binary",target_variable = "is_win",metrics = "auc",logger = None):
    logger = logger or logging.getLogger(__name__)
    logger.info("start training")

    df = df[df["ri_Year"] >= 2013]
    count = 0

    def __objective(values):
        nonlocal count
        count += 1
        msg = "start round {} with {}".format(count,values)
        print(msg)
        logging.info(msg)

        n_folds = 5 
        num_boost_round = 20000
        early_stopping_rounds = 200
        params = {
            'max_depth': values["max_depth"], 
            'num_leaves': values["num_leaves"],
            'min_child_samples': values["min_child_samples"], #minimal number of data in one leaf.
            'subsample': values["subsample"], #alias of bagging_fraction
            'colsample_bytree': values["colsample_bytree"],
            'lambda_l1' : values["lambda_l1"],
            'bagging_freq': 1,
            'max_delta_step' : 20,
            #'scale_pos_weight': 1,

            'learning_rate': 0.01,
            #'subsample_freq': 1,
            'metric':metrics,
            'boosting_type': 'gbdt',
            'objective': objective,
            'task':'train',
            'verbose':-1,
        }

        #kf = KFold(n = train_x.shape[0],n_folds = n_folds,random_state = 32,shuffle = True)
        kf = RaceKFold(df,n_folds)
     
        scores = []
        for i,(train,test) in enumerate(kf):
            x1,y1 = filter_df(train,train = True, target_variable = target_variable)
            x2,y2 = filter_df(test, target_variable = target_variable)

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
            log_msg  ="[{}] score ... {}".format(count,score)
            print(log_msg)
            logging.info(log_msg)

        avg_score = sum(scores)/len(scores)
        log_msg = "avg_score : {}".format(avg_score)
        print(log_msg)
        logging.info(log_msg)

        show_importance = False
        if show_importance:
            importance = zip(x1.columns,model.feature_importance())
            importance = sorted(importance, key = lambda x : x[1])
            for k,v in importance:
                print("{} : {}".format(k,v))
            score_threhold(model,test)
        return avg_score

        #return -avg_score
    paramaters = {
        "max_depth" : hp.choice("max_depth",range(10,200)),
        "num_leaves" : hp.choice("num_leaves",range(10,1000)),
        "min_child_samples" : hp.choice("min_child_samples",range(0,100)),
        "subsample" : hp.uniform("subsample",0.01,1.0),
        "colsample_bytree" : hp.uniform("colsample_bytree",0.01,1.0),
        "lambda_l1" : hp.uniform("lambda_l1",0,5.0),
    }
    trials = Trials()
    best = fmin(
        __objective,
        space = paramaters,
        algo=tpe.suggest,
        max_evals = 400,
        trials = trials)

    print(best)

    #plot_convergence(res_gp)

def label_encoding(df,categoricals = {},logger = None):
    logger = logger or logging.getLogger(__name__)
    logger.info("start label_encoding unit")

    categoricals = [c for c in df.columns if c in categoricals]
    for k in tqdm(categoricals,disable = TQDM_DISABLE):
        le = LabelEncoder()
        df[k] = le.fit_transform(df[k].astype(str))
    return df

def remove_unused(df,removes = [],logger = None):
    logger = logger or logging.getLogger(__name__)
    logger.info("start remove_unsed unit")

    columns = list(filter(lambda x : x not in removes,df.columns))
    return df.loc[:,columns]

def preprocess(df,logger = None):
    logger = logger or logging.getLogger(__name__)
    logger.info("start preprocss unit")

    #df.dropna(subset = ["hr_OrderOfFinish"],inplace = True)
    #drop newbies
    df = df.loc[~df.loc[:,"hi_ConditionClass"].isin(["0s","s"]),:]
    df = add_features.parse_feature(df)
    
    #df = df.loc[df.loc[:,"ri_CourseCode"] == "06s",:]
    df = df.loc[df.loc[:,"distance_category"]].isin(["sprint"])

    #add target variables
    df = calc_pop_order(df)
    df = utils.generate_target_features(df)

    #remove illegal values
    df["hi_RunningStyle"] = df.loc[:,"hi_RunningStyle"].where(df.loc[:,"hi_RunningStyle"] != "8s","s")

    print("add_runstyle_info")
    df = add_features.add_run_style_info(df)

    print("add_comb_info")
    df = add_features.add_combinational_feature(df)
    print(df.columns)

    print("add_change_info")
    df = add_features.add_change_info(df)

    print("add_datetime_info")
    df = add_features.add_datetime_info(df)

    print("add_surface_info")
    df = add_features.add_course_info(df)
    return df

def feature_engineering(df,numericals = [],categoricals = [],logger = None):

    logger = logger or logging.getLogger(__name__)
    logger.info("start feature_encoding unit")

    print("log margin")
    df["per1_log_delta"] = np.log(1 + df["pre1_TimeDelta"])
    df["per2_log_delta"] = np.log(1 + df["pre2_TimeDelta"])
    df["per3_log_delta"] = np.log(1 + df["pre3_TimeDelta"])

    print("add speed")
    df = add_features.add_speed(df)

    print("add normalized time")
    df = add_features.time_norm(df)

    print("add corner")
    df = add_features.add_corner_info(df)

    print("fix_extra_info")
    df = add_features.fix_extra_info(df)

    print("add_delta_info")
    df = add_features.add_delta_info(df)

    print("past averaginge")
    df = add_features.avg_past3(df,categoricals = categoricals)

    print("normalizing with race")
    df = add_features.norm_with_race(df,categoricals)

    print("target encoding")
    df = add_features.target_encoding(df,categoricals)
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

def filter_df(df,target_variable = "is_win",train = False):
    #target_variable = "pop_prediction"
    remove = ["hi_RaceID","ri_Year","hr_OrderOfFinish","hr_PaybackWin","hr_PaybackPlace",
            "is_win","is_place","win_return","place_return","li_WinOdds",
            "oof_over_pops","pop_prediction"]
    remove += PROCESS_FEATURE
    remove = list(set(remove))
    features = []
    for c in df.columns:
        should_add = True
        for r in remove:
            if c.startswith(r):
                should_add = False
        if should_add:
            features.append(c)
    features = sorted(features)
    if train:
        #df = upsampling(df,target_variable)
        pass
    y = df.loc[:,target_variable]
    #features = list(filter(lambda x : x not in remove, df.columns))
    return df.loc[:,features],y

def downsample(df,y):
    target = "hr_OrderOfFinish"
    x_columns = df.columns
    df = pd.concat([df,y],axis = 1)
    df_true = df[df[target] == 1]
    df_false = df[df[target] == 0].sample(n = len(df_true),random_state = 32)
    df = pd.concat([df_true,df_false],axis = 0)
    return df[x_columns],df[target]

def evaluate(df,model,features,show_results = True, mode = "argmax"):
    #df = pd.concat([df,y],axis = 1)
    x1,y1 = filter_df(df)
    df.loc[:,"pred"] = model.predict(x1)
    #contribute = model.predict(x1,pred_contrib = True)[0]
    df.loc[:,"pred_bin"] = 0

    groups = df.loc[:,["hi_RaceID","pred","pred_bin"]].groupby("hi_RaceID")
    for i,g in groups:
        if mode == "argmax":
            df.loc[g["pred"].idxmax(),"pred_bin"] = 1
        else:
            df.loc[g["pred"].idxmin(),"pred_bin"] = 1
    race_num = len(groups)

    df["pred_order"] = df.loc[:,["hi_RaceID","pred"]].groupby("hi_RaceID")["pred"].rank(method = "min")
    df["kaime"] = (df["pred_order"] < df["pops_order"]).astype(np.int8)

    win_hit = df.loc[:,"is_win"].values * df.loc[:,"pred_bin"].values
    win_hit_ratio = win_hit.sum()/race_num
    win_payback = df.loc[:,"hr_PaybackWin"].fillna(0).values * df.loc[:,"pred_bin"].values
    win_return = win_payback.sum() / (race_num * 100)

    kaime = df.loc[:,"kaime"].values * df.loc[:,"pred_bin"].values
    kaime_buy = kaime.sum()
    win_kaime_hit = (kaime * df.loc[:,"is_win"]).sum()
    win_kaime_payback = (kaime * df.loc[:,"hr_PaybackWin"].fillna(0)).sum()
    win_kaime_hit_ratio = win_kaime_hit / kaime_buy
    win_kaime_ret_ratio = win_kaime_payback / (kaime_buy * 100)

    place_hit = df.loc[:,"is_place"].values * df.loc[:,"pred_bin"].values
    place_hit_ratio = place_hit.sum()/race_num
    place_payback = df.loc[:,"hr_PaybackPlace"].fillna(0).values * df.loc[:,"pred_bin"].values
    place_return = place_payback.sum() / (race_num * 100)

    place_kaime_hit = (kaime * df.loc[:,"is_place"]).sum()
    place_kaime_payback = (kaime * df.loc[:,"hr_PaybackPlace"].fillna(0)).sum()
    place_kaime_hit_ratio = place_kaime_hit / kaime_buy
    place_kaime_ret_ratio = place_kaime_payback / (kaime_buy * 100)

    if show_results:
        print("        hit      ret")
        print("win     {:.3f},  {:.3f}".format(win_hit_ratio,win_return))
        print("win_k   {:.3f},  {:.3f}".format(win_kaime_hit_ratio,win_kaime_ret_ratio))
        print("place   {:.3f}   {:.3f}".format(place_hit_ratio,place_return))
        print("place_k {:.3f},  {:.3f}".format(place_kaime_hit_ratio,place_kaime_ret_ratio))
    result_dict = {
        "win_hit":"",
        "win_ret":"",
        "place_hit":"",
        "place_ret":"",
        "win_k_hit" : "",
        "win_k_ret" : "",
        "place_k_hit" :"",
        "place_k_ret" : "",

    }
    return win_hit_ratio,place_hit_ratio,win_return,place_return

def top1k(df):
    #is_hit = np.zeros(len(df))
    #is_hit = df["pred"].idxmax() == df["hr_OrderOfFinish"].idxmin()
    df.loc[df["pred"].idxmax(),"pred_bin"] = 1
    return df["pred_bin"]

def to_trainable(df,year,target = "hr_OrderOfFinish"):
    df = df[df["ri_Year"] >= 2013]

    test_size = 1000
    random_seed = 32
    np.random.seed(random_seed)
    race_id = df.loc[:,"hi_RaceID"].unique()
    np.random.shuffle(race_id)
    #train = df[df["ri_Year"] >= year]
    #test = df[df["ri_Year"] < year]
    test_ids = race_id[:test_size]
    train_ids = race_id[test_size:]
    train = df.loc[df["hi_RaceID"].isin(train_ids),:]
    test = df.loc[df["hi_RaceID"].isin(test_ids),:]
    del(df)
    gc.collect()
    return train,test

def add_target_variables(df):
    #df["is_win"] = df["hr_PaybackWin"].notnull().astype(np.int32)
    df["is_win"] = (df["hr_OrderOfFinish"] == 1).astype(np.int32)
    df["is_place"] = df["hr_PaybackPlace"].notnull().astype(np.int32)
    df["win_return"] = (df["hr_PaybackWin"].fillna(0) - 100) / 100
    df["place_return"] = (df["hr_PaybackWin"].fillna(0) - 100) / 100
    df["oof_over_pops"] = (df["hr_OrderOfFinish"] < df["hi_BasePops"]).astype(np.int32)
    df["pop_prediction"] = df["oof_over_pops"] * df["is_place"]
    return df

def RaceKFold(df,n):
    #random_seed = 32
    random_seed = 16
    np.random.seed(random_seed)
    race_id = df.loc[:,"hi_RaceID"].unique()
    np.random.shuffle(race_id)

    chunk_size = len(race_id)//n

    for i in range(n):
        start = i * chunk_size
        if i+1 != n:
            end = (i+1) * chunk_size
            target_id = race_id[start:end]
        else:
            target_id = race_id[start:]
        idx = df.loc[:,"hi_RaceID"].isin(target_id)
        train = df.loc[~idx,:]
        test = df.loc[idx,:]
        yield train,test

class Score():
    def __init__(self):
        self.win_hit = []
        self.place_hit = []
        self.win_return = []
        self.place_return = []

    def show(self):
        win_hit,win_hit_std = self.get_metrics("win_hit")
        place_hit,place_hit_std = self.get_metrics("place_hit")
        win_return,win_return_std = self.get_metrics("win_return")
        place_return,place_return_std = self.get_metrics("place_return")
        print("                   hit              ret")
        print("win   {:.3f} ({:.3f}),  {:.3f} ({:.3f})".format(win_hit,win_hit_std,win_return,win_return_std))
        print("place {:.3f} ({:.3f}),  {:.3f} ({:.3f})".format(place_hit,place_hit_std,place_return,place_return_std))

    def register(self,win_hit,place_hit,win_return,place_return):
        self.win_hit.append(win_hit)
        self.place_hit.append(place_hit)
        self.win_return.append(win_return)
        self.place_return.append(place_return)

    def get_metrics(self,target):
        metrics_dict = {
            "win_hit" : self.win_hit,
            "place_hit" : self.place_hit,
            "win_return" : self.win_return,
            "place_return" : self.place_return,
        }
        try:
            target_values = metrics_dict[target]
            mean = np.mean(target_values)
            std = np.std(target_values)
            return mean,std

        except  KeyError:
            raise Exception("unknown target : {}".format(target))

def output_predicted_dataset(df1,df2,model):
    print("outputing")
    ls = zip([df1,df2],["train_pred.csv","test_pred.csv"])
    for df,fname in ls:
        x1,y1 = filter_df(df)
        del(y1);gc.collect()
        df["pred"] = model.predict(x1)
        df.to_csv(fname)

def calc_pop_order(df):
    key = "li_WinOdds"
    if key not in df.columns:
        print("no key")
        return df
    df[key] = df.loc[:,key].where(df.loc[:,key] >= 1,99)
    df["pops_tmp"] = 1/df.loc[:,key]
    group = df.loc[:,["hi_RaceID","pops_tmp"]].groupby("hi_RaceID")
    for n,g in tqdm(group, disable = TQDM_DISABLE):
        g.sort_values(by = "pops_tmp",ascending = False,inplace = True)
        g.insert(0, 'pops_order', range(1,len(g)+1))
        df.loc[g.index,"pops_order"] = g.loc[:,"pops_order"]
    df.drop("pops_tmp",axis = 1,inplace = True)
    return df

def score_threhold(model,df):

    x1,y1 = filter_df(df)
    pred = model.predict(x1)
    mn = pred.min()
    mx = pred.max()

    print("max_value : {}".format(mx))
    print("min_value : {}".format(mn))

    plot_num = 50
    for i in range(0,plot_num):
        i = i/plot_num
        buy = np.greater(pred,i).astype(np.int8)
        win_hit = df.loc[:,"is_win"].values * buy
        win_ret = df.loc[:,"hr_PaybackWin"].fillna(0).values * buy
        win_hitper = win_hit.sum()/buy.sum()
        win_retper = win_ret.sum()/(buy.sum() * 100)
        place_hit = df.loc[:,"is_place"].values * buy
        place_ret = df.loc[:,"hr_PaybackPlace"].fillna(0).values * buy
        place_hitper = place_hit.sum()/buy.sum()
        place_retper = place_ret.sum()/(buy.sum() * 100)
        print("score : {} , buynum :  {}, - win : {}, {}, place : {} , {}".format(i,buy.sum(), win_hitper ,win_retper ,place_hitper, place_retper))

def upsampling(df,target):
    value_1 = df.loc[df.loc[:,target] == 1,:]
    value_0 = df.loc[df.loc[:,target] == 0,:]
    size_1 = len(value_1)
    size_0 = len(value_0)

    upsample_rate = size_0/size_1
    value_1 = value_1.sample(frac = upsample_rate,replace = True)
    df = pd.concat([value_0,value_1],axis = 0)
    del(value_1,value_0);gc.collect()
    df = df.sample(frac = 1.0).reset_index(drop = True)
    return df

def fillna(df, mode = "normal",categoricals = []):
    numericals = [c for c in df.columns if c not in categoricals]
    if mode == "normal":
        for c in numericals:
            mean = df.loc[:,c].mean()
            df.loc[:,c] = df.fillna(mean)
        return df

    elif mode == "zero":
        df.loc[:,numericals] = df.loc[:,numericals].fillna(0)
        return df

    elif mode == "race":
        targets = numericals + ["hi_RaceID"]
        groups = df.loc[:,targets].groupby("hi_RaceID")
        group_mean = groups.mean().reset_index()
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
            if c == "hi_RaceID" or c in PROCESS_FEATURE:
                continue
            mean_col = "{}_mean".format(c)
            if mean_col not in df.columns:
                continue
            df.loc[:,c] = df.loc[:,c].fillna(df.loc[:,mean_col])

        drop_targets = [c for c in mean_columns if c != "hi_RaceID"]
        df.drop(drop_targets,inplace = True,axis = 1)
        return df

if __name__ == "__main__":
    main()
