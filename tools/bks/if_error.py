import pandas as pd
import os
import numpy as np
if __name__=="__main__":
    path="work_dirs/yolov8_s_syncbn_fast_8xb16-500e_ykn/export-01/TPMBDBDL.csv"
    # path=os.path.join("work_dirs","mask2fromer_ykn","export-01","TPMBDBDL.csv")
    df=pd.read_csv(path)
    print(df['ImageID'])
    print(len(np.unique(np.array(df['ImageID']))))
    print(179-len(np.unique(np.array(df['ImageID']))))