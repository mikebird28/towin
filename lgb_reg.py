import lightgbm as lgb
import preprep
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split,KFold
from skopt.space import Real,Integer
from skopt.plots import plot_convergence
from skopt import gp_minimize
import numpy as np
import gc
import utils
from tqdm import tqdm

import add_features

PROCESS_FEATURE = ["hi_RaceID","ri_Year","hr_OrderOfFinish","li_WinOdds","margin"]

def main():
    mode = "normal"
    #mode = "tuning"

    target_variable = "margin"

    columns_dict = load_dtypes("./data/dtypes.csv")
    columns = sorted(columns_dict.keys())
    categoricals,numericals = cats_and_nums(columns,columns_dict)
    categoricals = categoricals + add_features.additional_categoricals()
    removes = load_removes("remove.csv")
    #removes = load_removes("remove_new.csv")

    df = pd.read_csv("./data/output.csv",dtype = columns_dict,usecols = columns)
    df = df.sort_index(axis=1, ascending=True)
    p = preprep.Preprep("./cache_files/lgb")
    p = p.add(preprocess,name = "preprocess",cache_format = "feather")
    p = p.add(utils.replace_few, params = {"categoricals" : categoricals} , cache_format = "feather", name ="replace_few")
    p = p.add(label_encoding,params = {"categoricals":categoricals},cache_format = "feather",name = "lenc")
    p = p.add(remove_unused,params = {"removes" : removes},cache_format = "feather",name = "remove")
    p = p.add(feature_engineering,params = {"numericals":numericals},cache_format = "feather",name = "feng")
    #p = p.add(add_target_variables,cache_format = "feather",name = "target")
    df = p.fit_gene(df,verbose = True)

    if mode == "normal":
        kf = RaceKFold(df,n = 10)
        score = Score()
        for i,(df1,df2) in enumerate(kf):
            if i == 0:
                model = light_gbm(df1,df2,categoricals,show_importance = True,score = score,target_variable = target_variable)
                output_predicted_dataset(df1,df2,model)
            else:
                light_gbm(df1,df2,categoricals,show_importance = False,score = score)

            del(df1,df2);gc.collect()
        score.show()
    else:
        optimize_params(df,categoricals)

    #df1,df2 = to_trainable(df, year = 2017)
    #del(p,df);gc.collect()

    #df_ds,y_ds = downsample(df,y)
    #light_gbm(df1,df2,categoricals)

    #features = list(filter(lambda x : x not in remove, x.columns))
    #evaluate(df2,model,features)

def search_unnecessary(fdf,categoricals):
    score = light_gbm(fdf,categoricals)
    for col in fdf.columns:
        feat_except_col = [c for c in fdf.columns if c != col]
        cat_except_col = [c for c in categoricals if c != col]
        s = light_gbm(fdf.loc[:,feat_except_col],y,cat_except_col,show_importance = False)
        typ = "T"
        if s < score:
            typ = "F"
        print("{} - score except  {} is  {}".format(typ,col,s))


def light_gbm(df1,df2,categoricals,score = None,show_importance = True,typ = "classify",target_variable = "is_win"):
    x1,y1 = filter_df(df1,target_variable)
    x2,y2 = filter_df(df2)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',

        'max_depth': 3,
        'num_leaves': 500,
        'min_child_samples': 50,
        'scale_pos_weight': 1.0,
        'subsample': 0.28,
        'colsample_bytree': 0.789,
        'bagging_freq': 5,
        'learning_rate': 0.005,
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
            print("{} : {}".format(k,v))

    win_hit,place_hit,win_return,place_return = evaluate(df2,model,features,show_results = True)
    score_threhold(model,df2)
    if score is not None:
        score.register(win_hit,place_hit,win_return,place_return)
    return model

def optimize_params(df,categoricals):
    #df = df[df["ri_Year"] >= 2013]
    metrics = "auc"
    count = 0

    def __objective(values):
        nonlocal count
        count += 1
        n_folds = 5 
        num_boost_round = 20000
        early_stopping_rounds = 200
        print(values)
        params = {
            'max_depth': values[0], 
            'num_leaves': values[1],
            'min_child_samples': values[2], #minimal number of data in one leaf.
            'subsample': values[3], #alias of bagging_fraction
            'colsample_bytree': values[4],
            'lambda_l1' : values[5],
            'bagging_freq': 1,
            'max_delta_step' : 20,
            'scale_pos_weight': 1,

            'learning_rate': values[6],
            #'subsample_freq': 1,
            'metric':metrics,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'task':'train',
            'verbose':-1,
        }

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
        Integer(3, 50, name='max_depth'),
        Integer(100,1000, name='num_leaves'),
        Integer(70, 100, name='min_child_samples'), #1
        Real(0.01, 1.0, name='subsample'),
        Real(0.01, 1.0, name='colsample_bytree'),
        Real(0, 20.0,name='lambda_l1'),
        Real(0.001, 0.01, name='learning_rate'),
    ]
    res_gp = gp_minimize(__objective, space, n_calls=700,n_random_starts=1,xi = 0.005)
    print(res_gp)
    plot_convergence(res_gp)

def label_encoding(df,categoricals = {}):
    categoricals = [c for c in df.columns if c in categoricals]
    for k in tqdm(categoricals):
        le = LabelEncoder()
        df[k] = le.fit_transform(df[k].astype(str))
    return df

def remove_unused(df,removes = []):
    columns = list(filter(lambda x : x not in removes,df.columns))
    return df.loc[:,columns]

def preprocess(df):
    #df.dropna(subset = ["hr_OrderOfFinish"],inplace = True)
    #drop newbies
    df = df.loc[~df.loc[:,"hi_ConditionClass"].isin(["0s","s"]),:]
    df = df.loc[df.loc[:,"ri_CourseCode"] == "06s",:]

    df = calc_pop_order(df)
    #add target variables
    df.loc[:,"is_win"] = (df.loc[:,"hr_OrderOfFinish"] == 1).astype(np.int32)
    df.loc[:,"is_place"] = df.loc[:,"hr_PaybackPlace"].notnull().astype(np.int32)
    df.loc[:,"win_return"] = (df.loc[:,"hr_PaybackWin"].fillna(0) - 100) / 100
    df.loc[:,"place_return"] = (df.loc[:,"hr_PaybackWin"].fillna(0) - 100) / 100
    df.loc[:,"margin"] = df.loc[:,"hr_FinishingDelta"]
    df.loc[:,"oof_over_pops"] = (df["hr_OrderOfFinish"] < df["pops_order"]).astype(np.int32)
    df.loc[:,"pop_prediction"] = df["oof_over_pops"] * df["is_place"]

    #remove illegal values
    df["hi_RunningStyle"] = df.loc[:,"hi_RunningStyle"].where(df.loc[:,"hi_RunningStyle"] != "8s","s")
    df = add_features.parse_feature(df)

    print("add_runstyle_info")
    df = add_features.add_run_style_info(df)

    print("add_comb_info")
    df = add_features.add_combinational_feature(df)

    print("add_change_info")
    df = add_features.add_change_info(df)

    print("add_datetime_info")
    df = add_features.add_datetime_info(df)

    print("add_surface_info")
    df = add_features.add_course_info(df)
    return df

def feature_engineering(df,numericals = []):
    print("add speed")
    df = add_features.add_speed(df)

    print("add corner")
    df = add_features.add_corner_info(df)

    print("fix_extra_info")
    df = add_features.fix_extra_info(df)

    print("add_delta_info")
    df = add_features.add_delta_info(df)

    print("normalizing with race")
    df = add_features.norm_with_race(df,numericals)

    #print("add_race_avg")
    #df = add_features.add_race_mean(df,numericals)
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

def load_dtypes(dtypes_path):
    dtypes = {}
    allow_hr = ["hr_OrderOfFinish","hr_PaybackWin","hr_PaybackPlace"]
    with open(dtypes_path) as fp:
        for row in fp.readlines():
            row_s = row.strip().split(",")
            name = row_s[0]
            dtyp = row_s[1]
            if name.startswith("hr_") and name not in allow_hr:
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

def filter_df(df,target_variable = "is_win"):
    #target_variable = "pop_prediction"
    y = df.loc[:,target_variable]
    remove = ["hi_RaceID","ri_Year","hr_OrderOfFinish","hr_PaybackWin","hr_PaybackPlace",
            "is_win","is_place","win_return","place_return","li_WinOdds",
            "oof_over_pops","pop_prediction"]
    features = []
    for c in df.columns:
        should_add = True
        for r in remove:
            if c.startswith(r):
                should_add = False
        if should_add:
            features.append(c)
    features = sorted(features)

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

def evaluate(df,model,features,show_results = True,typ = "argmax"):
    #df = pd.concat([df,y],axis = 1)
    x1,y1 = filter_df(df)
    df.loc[:,"pred"] = model.predict(x1)
    df.loc[:,"pred_bin"] = 0

    groups = df.loc[:,["hi_RaceID","pred","pred_bin"]].groupby("hi_RaceID")
    for i,g in groups:
        if typ == "argmax":
            df.loc[g["pred"].idxmax(),"pred_bin"] = 1
        else:
            df.loc[g["pred"].idxmin(),"pred_bin"] = 1
    race_num = len(groups)

    win_hit = df.loc[:,"is_win"].values * df.loc[:,"pred_bin"].values
    win_hit_ratio = win_hit.sum()/race_num
    win_payback = df.loc[:,"hr_PaybackWin"].fillna(0).values * df.loc[:,"pred_bin"].values
    win_return = win_payback.sum() / (race_num * 100)

    place_hit = df.loc[:,"is_place"].values * df.loc[:,"pred_bin"].values
    place_hit_ratio = place_hit.sum()/race_num
    place_payback = df.loc[:,"hr_PaybackPlace"].fillna(0).values * df.loc[:,"pred_bin"].values
    place_return = place_payback.sum() / (race_num * 100)

    if show_results:
        print("        hit        ret")
        print("win   {:.3f},  {:.3f}".format(win_hit_ratio,win_return))
        print("place {:.3f}   {:.3f}".format(place_hit_ratio,place_return))
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
    random_seed = 32
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
    for n,g in tqdm(group):
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

if __name__ == "__main__":
    main()
