
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc

def add_race_mean(df,numericals):
    key = ["hi_RaceID"]
    remove = ["hr_OrderOfFinish","hr_PaybackWin","hr_PaybackPlace","is_win","is_place","win_return","place_return","ri_Year","hi_Prize","ri_Distance"
            "li_FiledStatus","hi_Times","ri_FirstPrize","ri_SecondPrize","ri_ThirdPrize","ri_HaedCount","ri_Month"]
    
    numericals = [c for c in df.columns if c in numericals]
    #remove process features
    numericals = [c for c in numericals if c not in remove]

    targets = numericals + [k for k in key if k not in numericals]
    group = df[targets].groupby(key)
    group_mean = group.mean().reset_index()
    mean_columns = []
    for c in group_mean.columns:
        if c in key:
            mean_columns.append(c)
        else:
            mean_columns.append("{}_mean".format(c))
    group_mean.columns = mean_columns
    df = df.merge(group_mean,on = key, how = "left")
    for c in targets:
        mean_col = "{}_mean".format(c)
        if mean_col not in df.columns:
            continue
        df.loc[:,c] = df.loc[:,c] - df.loc[:,mean_col]
    return df

def norm_with_race(df,numericals):
    key = ["hi_RaceID"]
    remove = ["hr_OrderOfFinish","hr_PaybackWin","hr_PaybackPlace","is_win","is_place","win_return","place_return","ri_Year","hi_Prize","ri_Distance"
            "li_FiledStatus","hi_Times","ri_FirstPrize","ri_SecondPrize","ri_ThirdPrize","ri_HaedCount","ri_Month"]

    numericals = [c for c in df.columns if c in numericals]
    numericals = [c for c in numericals if c not in remove]
    targets = numericals + [k for k in key if k not in numericals]

    group = df[targets].groupby(key)
    group_mean = group.mean().reset_index()
    group_std = group.std().reset_index()
    group_std.clip(lower = 1e-5, inplace = True)

    norm_columns = []
    std_columns = []
    for c in group_mean.columns:
        if c in key:
            norm_columns.append(c)
            std_columns.append(c)
        else:
            norm_columns.append("{}_norm".format(c))
            std_columns.append("{}_std".format(c))
            #df.loc[df.loc[:,c] == 0.0,c] = 1.0

    group_mean.columns = norm_columns
    group_std.columns = std_columns

    df = df.merge(group_mean,on = key, how = "left")
    df = df.merge(group_std,on = key, how = "left")

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
        "JockeyBrinker"
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
        df.loc[:,total] = df.loc[:,first] + df.loc[:,second] + df.loc[:,third] + df.loc[:,lose]
        df.loc[:,place] = df.loc[:,first] + df.loc[:,second] + df.loc[:,third]
        df.loc[:,win_per] = df.loc[:,first]/df.loc[:,total]
        df.loc[:,place_per] = df.loc[:,place]/df.loc[:,total]

        df.drop(second,inplace = True,axis = 1)
        df.drop(third,inplace = True,axis = 1)

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
        date1 = pd.to_datetime(df.loc[:,f1],format ="%Y%m%ds")
        date2 = pd.to_datetime(df.loc[:,f2],format ="%Y%m%ds")
        df.loc[:,name] = (date1 - date2).astype(np.int64)/divider
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

