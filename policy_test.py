
import pandas as pd
import utils,policy

DTYPE_PATH = "./data/dtypes.csv"

def main():
    df = pd.read_feather("./cache_files/fpolicy/drop.feather")
    df.drop("index",axis = 1, inplace = True)
    df,test_df = policy.split_with_year(df,year = 2017)

    columns_dict = utils.load_dtypes(DTYPE_PATH)
    columns = sorted(columns_dict.keys())
    categoricals,numericals = utils.cats_and_nums(columns,columns_dict)
    policy.policy_gradient(df,test_df,categoricals)

if __name__ == "__main__":
    main()
