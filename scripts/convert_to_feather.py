#!/usr/bin/env python

import pandas as pd
import sys

def main(input_path,output_path):
    print("convert {} => {}".format(input_path,output_path))
    df = pd.read_csv(input_path)
    df.to_feather(output_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("error : number of argument is wrong")
        sys.exit(1)
    main(sys.argv[1],sys.argv[2])
