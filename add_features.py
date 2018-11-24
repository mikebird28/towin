
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc

def parse_feature(df):
    def to_dist_cat(x):
        if 1000 <= x and x <= 1400:
            return "sprint"
        elif 1400 < x and x <= 1600:
            return "mile"
        elif 1600 < x and x <= 2200:
            return "middle"
        elif 2200 < x and x < 2600:
            return "classic"
        elif 2600 <= x:
            return "stayer"

    def parse_hoof_size(x):
        large = ["01s","05s","09s","17s","21s"]
        midium = ["02s","06s","10s","18s","22s"]
        small = ["03s","07s","11s","19s","23s"]
        quite_small = ["04s","08s","12s","20s","24s"]
        if x in large:
            return "l"
        elif x in midium:
            return "m"
        elif x in small:
            return "s"
        elif x in quite_small:
            return "qs"
        else:
            return "o"

    def parse_hoof_type(x):
        t1 = ["01s","02s","03s","04s"]
        t2 = ["05s","06s","07s","08s"]
        t3 = ["09s","10s","11s","12s"]
        t4 = ["17s","18s","19s","20s"]
        t5 = ["21s","22s","23s","24s"]
        if x in t1:
            return "t1"
        elif x in t2:
            return "t2"
        elif x in t3:
            return "t3"
        elif x in t4:
            return "t4"
        elif x in t5:
            return "t5"
        else:
            return "o"

    df["distance_category"] = df.loc[:,"ri_Distance"].apply(to_dist_cat)
    df["hoof_type"] = df.loc[:,"hi_HoofCode"].apply(parse_hoof_type)
    df["hoof_size"] = df.loc[:,"hi_HoofCode"].apply(parse_hoof_size)
    df.drop("hi_HoofCode",axis = 1,inplace  = True)
    return df

def norm_with_race(df,categoricals = []):
    key = ["hi_RaceID"]
    remove = ["hr_OrderOfFinish","hr_PaybackWin","hr_PaybackPlace","hr_TimeDelta","hr_FinishingTime",
            "is_win","is_place","win_return","place_return","margin","norm_time","pops_order",
            "ri_Year","hi_Prize","ri_Distance","li_FiledStatus","hi_Times","ri_FirstPrize","ri_SecondPrize","ri_ThirdPrize","ri_HaedCount","ri_Month",
            "hr_TimeDelta","hr_FinishingTime","margin","norm_time","win_return","place_return","oof_over_pops","pop_prediction",
    ]

    numericals = [c for c in df.columns if c not in categoricals]
    numericals = [c for c in numericals if c not in remove]
    targets = numericals + [k for k in key if k not in numericals]

    group = df[targets].groupby(key)
    group_mean = group.mean().reset_index()
    group_std = group.std().reset_index()
    #group_min = group.min().reset_index()
    #group_max = group.max().reset_index()
    group_std.clip(lower = 1e-5, inplace = True)

    norm_columns = []
    std_columns = []
    #min_columns = []
    #max_columns = []
    for c in group_mean.columns:
        if c in key:
            norm_columns.append(c)
            std_columns.append(c)
            #min_columns.append(c)
            #max_columns.append(c)
        else:
            norm_columns.append("{}_norm".format(c))
            std_columns.append("{}_std".format(c))
            #min_columns.append("{}_min".format(c))
            #cmax_columns.append("{}_max".format(c))
            #df.loc[df.loc[:,c] == 0.0,c] = 1.0

    group_mean.columns = norm_columns
    group_std.columns = std_columns
    #group_min.columns = min_columns
    #group_max.columns = max_columns

    df = df.merge(group_mean,on = key, how = "left")
    df = df.merge(group_std,on = key, how = "left")
    #df = df.merge(group_min,on = key, how = "left")
    #df = df.merge(group_max,on = key, how = "left")

    for c in targets:
        if c == "hi_RaceID":
            continue
        norm_col = "{}_norm".format(c)
        std_col = "{}_std".format(c)

        if (norm_col not in df.columns) or (std_col not in df.columns):
            continue
        df.loc[:,norm_col] = (df.loc[:,c] - df.loc[:,norm_col])/df.loc[:,std_col]

    drop_targets = [c for c in std_columns if c != "hi_RaceID"]
    df.drop(drop_targets,inplace = True,axis = 1)
    return df

def fix_extra_info(df):
    table_prefix = "ei"
    prefix_ls = [
        "Jra","Inter","Other","Surf","SurfDist","Dist","Rotation","Course","Jockey",
        "SurfaceGood","SurfaceMiddle","SurfaceBad","SlowPace","MiddlePace","HighPace",
        "Season","Frame","JockeyDist","JockeyTrack","JockeyTrainer","JockeyBrinker","JockeyOwner","TrainerOwner",
        "JockeyBrinker","JockeyTrackDist",
    ]
    for p in tqdm(prefix_ls):
        #existing features"
        first = "{}_{}{}".format(table_prefix,p,"First")
        second = "{}_{}{}".format(table_prefix,p,"Second")
        third = "{}_{}{}".format(table_prefix,p,"Third")
        lose = "{}_{}{}".format(table_prefix,p,"Lose")

        cancel_flag = False
        for c in [first,second,third,lose]:
            if c not in df.columns:
                cancel_flag = True
        if cancel_flag:
            continue

        #additional feature
        total = "{}_{}{}".format(table_prefix,p,"Total")
        place = "{}_{}{}".format(table_prefix,p,"Place")
        win_per = "{}_{}{}".format(table_prefix,p,"WinPer")
        place_per = "{}_{}{}".format(table_prefix,p,"PlacePer")
        lose_per = "{}_{}{}".format(table_prefix,p,"LosePer")

        df.loc[:,total] = df.loc[:,first] + df.loc[:,second] + df.loc[:,third] + df.loc[:,lose]
        df.loc[:,place] = df.loc[:,first] + df.loc[:,second] + df.loc[:,third]
        df.loc[:,win_per] = df.loc[:,first]/df.loc[:,total]
        df.loc[:,place_per] = df.loc[:,place]/df.loc[:,total]
        df.loc[:,lose_per] = df.loc[:,lose]/df.loc[:,total]

        df.drop(second,inplace = True,axis = 1)
        df.drop(third,inplace = True,axis = 1)
        #df.drop(lose,inplace = True,axis = 1)

    total = "ei_TotalTotal"
    win = "ei_TotalWin"
    place = "ei_TotalPlace"
    win_per = "ei_TotalWinPer"
    place_per = "ei_TotalPlacePer"
    df.loc[:,total] = df.loc[:,"ei_JraTotal"] + df.loc[:,"ei_InterTotal"] + df.loc[:,"ei_OtherTotal"]
    df.loc[:,win] = df.loc[:,"ei_JraFirst"] + df.loc[:,"ei_InterFirst"] + df.loc[:,"ei_OtherFirst"]
    df.loc[:,place] = df.loc[:,"ei_JraPlace"] + df.loc[:,"ei_InterPlace"] + df.loc[:,"ei_OtherPlace"]
    df.loc[:,win_per] = df.loc[:,win]/df.loc[:,total]
    df.loc[:,place_per] = df.loc[:,place]/df.loc[:,total]

    drop_targets = ["Jra","Inter","Other"]
    drop_columns = []
    for r in drop_targets:
        prefix = "{}_{}".format(table_prefix,r)
        drop_columns.append(prefix + "Total")
        drop_columns.append(prefix + "First")
        drop_columns.append(prefix + "Place")
        drop_columns.append(prefix + "WinPer")
        drop_columns.append(prefix + "PlacePer")
    df.drop(drop_columns,axis = 1,inplace = True)
    return df

def add_change_info(df):
    #(feature1,feature2,new_name)
    target_pairs = [
        ("pre1_Discipline","ri_Discipline","discipline_change"),
        ("pre1_RaceGrade","ri_RaceGrade","grade_change"),
        ("pre1_InOut","ri_InOut","inout_change"),
        ("pre1_LeftRight","ri_LeftrRight","leftright_change"),
        ("pre1_CourseCode","ri_CourseCode","course_change"),
    ]
    for pair in tqdm(target_pairs):
        f1 = pair[0]
        f2 = pair[1]
        name = pair[2]
        df.loc[:,name] = (df.loc[:,f1] == df.loc[:,f2]).astype(np.int8)
    return df

def add_delta_info(df):
    target_pairs = [
        ("li_HorseWeight","pre1_Weight","weight_delta"),
        ("pre1_Weight","pre2_Weight","weight_delta_pre1"),
        ("pre2_Weight","pre3_Weight","weight_delta_pre2"),
        ("li_BasisWeight","pre1_BasisWeight","basis_weight_delta"),
        ("pre1_Pass4","pre1_Pass3","pass_delta_pre1"),
        ("pre2_Pass4","pre2_Pass3","pass_delta_pre2"),
        ("pre3_Pass4","pre3_Pass3","pass_delta_pre3"),
        ("ri_Distance","pre1_Distance","distance_delta"),
        ("pre1_Distance","pre2_Distance","distance_delta_1_2"),
        ("pre2_Distance","pre3_Distance","distance_delta_2_3"),
        ("pre1_First3FTime","pre1_Last3FTime","first_last_delta_pre1"),
        ("pre2_First3FTime","pre2_Last3FTime","first_last_delta_pre2"),
        ("pre3_First3FTime","pre3_Last3FTime","first_last_delta_pre3"),
    ]
    for pair in tqdm(target_pairs):
        f1 = pair[0]
        f2 = pair[1]
        name = pair[2]
        df.loc[:,name] = df.loc[:,f1] - df.loc[:,f2]
    return df


def add_datetime_info(df):
    divider =  864  * 1e+11
    target_pairs = [
        ("ri_Datetime","pre1_RegisterdDate","elapsed_time"),
        ("pre1_RegisterdDate","pre2_RegisterdDate","elapsed_time_1_2"),
        ("pre2_RegisterdDate","pre3_RegisterdDate","elapsed_time_2_3"),
    ]
    for pair in tqdm(target_pairs):
        f1 = pair[0]
        f2 = pair[1]
        name = pair[2]
        date1 = pd.to_datetime(df.loc[:,f1],format ="%Y%m%ds",errors = "coerce")
        date2 = pd.to_datetime(df.loc[:,f2],format ="%Y%m%ds")
        df.loc[:,name] = (date1 - date2).astype(np.int64)/divider
        df.loc[:,name] = df.loc[:,name].where(df.loc[:,name] >= 0, np.nan)
    df.drop(["ri_Datetime","pre1_RegisterdDate","pre2_RegisterdDate","pre3_RegisterdDate"],inplace = True,axis = 1)
    return df

def add_speed(df):
    targets = ["pre1","pre2","pre3"]
    for t in targets:
        speed = t + "_Speed"
        finishing_time = t + "_" + "FinishingTime"
        distance = t + "_" + "Distance"
        if (not finishing_time in df.columns) or (not distance in df.columns):
            continue
        df.loc[:,speed] = df.loc[:,finishing_time]/df.loc[:,distance]

    for t in targets:
        oof_ratio = t + "_OofRatio"
        oof = t + "_OrderOfFinish"
        head_count = t + "_HeadCount"
        if (not oof in df.columns) or (not head_count in df.columns):
            continue
        df.loc[:,oof_ratio] = df.loc[:,oof]/df.loc[:,head_count]
    return df

def add_corner_info(df):
    targets = ["pre1","pre2","pre3"]
    for t in targets:
        first_corner_key = "first_corner_pass_{}".format(t)
        last_corner_key = "{}_Pass4".format(t)
        lf_delta = "first_last_pass_delta_{}".format(t)
        df[first_corner_key] = np.nan

        for i in [3,2,1]:
            key = "{}_Pass{}".format(t,i)
            df[first_corner_key] = df[key].where(~df.loc[:,key].isin([np.nan,0]),other = df[first_corner_key])
        df[lf_delta] = df.loc[:,last_corner_key] - df.loc[:,first_corner_key]

    return df

def add_course_info(df):
    def father_surf_winper(df):
        discipline = df["ri_Discipline"]
        if discipline == "1s":
            return df["ei_FatherTurfQuinellaPer"]
        elif discipline == "2s":
            return df["ei_FatherDirtQuinellaPer"]
        else:
            return np.nan
    def mother_surf_winper(df):
        discipline = df["ri_Discipline"]
        if discipline == "1s":
            return df["ei_MotherTurfQuinellaPer"]
        elif discipline == "2s":
            return df["ei_MotherDirtQuinellaPer"]
        else:
            return np.nan
    df["father_surf_winper"] = df.apply(father_surf_winper,axis = 1)
    df["mother_surf_winper"] = df.apply(mother_surf_winper,axis = 1)
    return df

horse_f = [
    "bi_FPedigreeCode2","bi_MFPedigreeCode2","hoof_size","hoof_type","hi_RunningStyle",
#    "hi_BodyType","hi_BellySize","hi_HipSize","hi_ChestSize",
]
race_f = ["ri_Discipline","li_FieldCode","distance_category","li_WeatherCode","ri_InOut",]

all_f = horse_f + race_f


def add_combinational_feature(df):
    def comb_f(df,new,f1,f2):
        df[new] = df.loc[:,f1] + df.loc[:,f2]
        df[new] = df[new].where(df[f1] != "s", "s")
        df[new] = df[new].where(df[f2] != "s", "s")
        return df

    for f1 in horse_f:
        for f2 in race_f:
            #premove prefix
            if f1 != f1.lower():
                f1_rem = f1.split("_")[1].lower()
            else:
                f1_rem = f1
            if f2 != f2.lower():
                f2_rem = f2.split("_")[1].lower()
            else:
                f2_rem = f2
            nf = "{}_{}".format(f1_rem,f2_rem)
            comb_f(df,nf,f1,f2)
    return df

def additional_categoricals():
    ls = ["hoof_size","hoof_type","distance_category","grade_change","inout_change","leftright_change","course_change"]

    for f1 in horse_f:
        for f2 in race_f:
            #premove prefix
            if f1 != f1.lower():
                f1_rem = f1.split("_")[1].lower()
            else:
                f1_rem = f1
            if f2 != f2.lower():
                f2_rem = f2.split("_")[1].lower()
            else:
                f2_rem = f2
            nf = "{}_{}".format(f1_rem,f2_rem)
            ls.append(nf)
    return ls

def add_run_style_info(df):
    #1 逃げ 2 先行 3 差し 4 追込 5 好位差し 6 自在
    groups = df[["hi_RaceID","hi_RunningStyle"]].groupby("hi_RaceID")
    counts_np = np.zeros([len(groups),6])
    rid_ls = []
    for i,(rid,group) in tqdm(enumerate(groups)):
        head_counts = len(group)
        rid_ls.append(rid)
        nige = group.loc[group.loc[:,"hi_RunningStyle"] == "1s","hi_RunningStyle"].count()
        senko = group.loc[group.loc[:,"hi_RunningStyle"] == "2s","hi_RunningStyle"].count()
        sasi = group.loc[group.loc[:,"hi_RunningStyle"] == "3s","hi_RunningStyle"].count()
        oikomi = group.loc[group.loc[:,"hi_RunningStyle"] == "4s","hi_RunningStyle"].count()
        koui = group.loc[group.loc[:,"hi_RunningStyle"] == "5s","hi_RunningStyle"].count()
        jizai = group.loc[group.loc[:,"hi_RunningStyle"] == "6s","hi_RunningStyle"].count()
        counts_np[i,0] = nige/head_counts
        counts_np[i,1] = senko/head_counts
        counts_np[i,2] = sasi/head_counts
        counts_np[i,3] = oikomi/head_counts
        counts_np[i,4] = koui/head_counts
        counts_np[i,5] = jizai/head_counts
    run_df =  pd.DataFrame(counts_np,columns = ["nige_ratio","senko_ratio","sasi_ratio","oikomi_ratio","kouji_ratio","jizai_ratio"])
    del(counts_np);gc.collect();
    run_df.index = rid_ls
    df = df.merge(run_df,left_on = "hi_RaceID",right_index = True)
    return df

def target_encoding(df,cats):
    race_info = ["ri_Discipline","distance_category","li_FieldCode","li_WeatherCode","hi_FrameNumber"]
    horse_info = ["hi_RunningType","bi_MFPedigreeCode2","bi_FPedigreeCode2","hoof_type","hoof_size","hi_RunningStyle","hi_SexCode","bi_FatherName"]
    prefixes = ["ri","hi","hr","li"]

    for ri in tqdm(race_info):
        for hi in horse_info:
            if (ri not in df.columns) or (hi not in df.columns):
                continue
            ri_split = ri.split("_")
            if ri_split[0] in prefixes:
                ri_name = "_".join(ri_split[1:]).lower()
            else:
                ri_name = ri.lower()
            hi_split = hi.split("_")
            if hi_split[0] in prefixes:
                hi_name = "_".join(hi_split[1:]).lower()
            else:
                hi_name = hi.lower()
            tmp_key = ri_name + hi_name + "_tmps"
            te_key = ri_name + "_" + hi_name + "_te"
            df[tmp_key] = df[hi].astype(str) + "_" +df[ri].astype(str)
            df = _target_encode(df,tmp_key,"is_place",new_key = te_key)
            df = _target_encode(df,tmp_key,"margin",new_key = te_key)
            df.drop(tmp_key,axis = 1,inplace = True)
    return df
 
def _target_encode(df,key,target,new_key = None,k = 500, f = 30, smoothing = True):
    #k (int) : minimum samples to take category average into account
    #f (int) : smoothing effect to balance categorical average vs prior

    if new_key is None:
        new_key = key + "_te"
    group = df[[key,target]].groupby(key)
    counts = group[target].mean().reset_index()
    counts.columns = [key,new_key]

    if smoothing:
        total_ratio = df.loc[:,target].mean()
        size = group[target].size().reset_index()
        size.columns = [key,"n_i"]
        counts = counts.merge(size,on = key).reset_index()
        counts["lambda"] = 1/(1 + np.exp(-(counts["n_i"]-k)/f))
        counts[new_key] = counts["lambda"] * counts[new_key] + (1 - counts["lambda"]) * total_ratio

    df = df.merge(counts.loc[:,[key,new_key]],on = key)
    return df

def time_norm(df):
    raw_targets = [
        (["Distance"],["TimeDelta","FinishingTime"]),
        (["Distance","Discipline"],["TimeDelta","FinishingTime"]),
        (["Distance","FieldStatus"],["TimeDelta","FinishingTime"]),
        (["Distance","CourseCode"],["TimeDelta","FinishingTime"]),
    ]

    targets = []
    drops = []
    for k,t in raw_targets:
        for i in range(3):
            pre_i_k = ["pre{}_{}".format(i+1,c) for c in k]
            pre_i_t = ["pre{}_{}".format(i+1,c) for c in t]
            tmp_key = "pre{}_{}_tmp".format(i+1,"_".join(k))
            df[tmp_key] = df.loc[:,pre_i_k].astype(str).apply("_".join,axis = 1)
            drops.append(tmp_key)
            targets.append((tmp_key,pre_i_t))

    for k,t in tqdm(targets):
        df = _time_norm(df,k,t)
    df.drop(drops,axis = 1, inplace = True)
    return df

def _time_norm(df,key,targets):
    #keys (str) : list of groupby key
    #targets (list) : list of target values
    group = df.loc[:, targets + [key]].groupby(key)
    mean = group.mean()
    std = group.std().clip(1e-8,None)

    mean.columns = ["{}_m".format(c) for c in mean.columns]
    std.columns = ["{}_s".format(c) for c in std.columns]

    agg = mean.join(std).reset_index()
    del(group,mean,std);gc.collect();

    df = df.merge(agg,on = key)
    drops = []
    for t in targets:
        prefix = extract_prefix(key)
        new_key = prefix + "_" + "_".join(drop_prefix([t,key])) + "_score"
        key_mean = t + "_m"
        key_std = t + "_s"
        drops.append(key_mean)
        drops.append(key_std)
        df[new_key] = (df[t] - df[key_mean])/df[key_std]
    df.drop(drops,axis = 1, inplace = True)
    return df

def extract_prefix(x):
    return x.split("_")[0]

def drop_prefix(x_ls):
    ls = []
    for i in x_ls:
        new_i = "_".join(i.split("_")[1:])
        ls.append(new_i)
    return ls


def avg_past3(df,categoricals = []):
    features = []
    for c in df.columns:
        if c in categoricals:
            continue
        if c.startswith("pre1_"):
            prefix_removed = "_".join(c.split("_")[1:])
            features.append(prefix_removed)
    drop_targets = []
    for c in features:
        past3_f = []
        for i in range(3):
            col = "pre{}_{}".format(i+1,c)
            past3_f.append(col)
            if i != 0:
                drop_targets.append(col)
        avg3_name = "avg3_{}".format(c)
        max3_name = "max3_{}".format(c)
        min3_name = "min3_{}".format(c)
        df.loc[:,avg3_name] = df.loc[:,past3_f].mean(axis = 1)
        df.loc[:,min3_name] = df.loc[:,past3_f].min(axis = 1)
        df.loc[:,max3_name] = df.loc[:,past3_f].max(axis = 1)
    df.drop(drop_targets, axis = 1, inplace = True)
    return df

def add_null_count(df,name = "null_count"):
    nullcount = df.isnull().astype(np.int8).sum(axis = 1)
    nullcount.name = name
    df = pd.concat([df,nullcount],axis = 1)
    return df
            
if __name__ == "__main__":
    df = pd.read_csv("data/output.csv",nrows = 50000)
    df["is_win"] = (df["hr_OrderOfFinish"] == 1).astype(np.int8)
    df = add_combinational_feature(df,[])
    _target_encode(df,"fpedigreecode_discipline","is_win")
    _target_encode(df,"fpedigreecode_field","is_win")
