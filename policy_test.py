
import pandas as pd
import utils,policy

DTYPE_PATH = "./data/dtypes.csv"

def main():
    df = pd.read_feather("./cache_files/fpolicy/drop.feather")
    df = df[df["ri_Year"] > 2016]
    test_df = df[df["ri_Year"] > 2017]

    columns_dict = utils.load_dtypes(DTYPE_PATH)
    columns = sorted(columns_dict.keys())
    categoricals,numericals = utils.cats_and_nums(columns,columns_dict)

    policy.policy_gradient(df,test_df,categoricals)

if __name__ == "__main__":
    main()
