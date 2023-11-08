import pandas as pd
import numpy as np

from fredapi import Fred
from utils.key import APIKEY as API_FRED


fred = Fred(api_key=API_FRED)

USREC = fred.get_series("USREC")
USREC=USREC.reset_index().rename(columns={'index':"date",0:"USREC"})
USREC['date'] = pd.to_datetime(USREC['date'])


def get_nonNANs_fullsample(frequency_dataset='monthly',add_USREC=True,max_date='2023-08-01',min_date=None):

    if frequency_dataset in ['monthly','quarterly'] == False:
        raise ValueError("Invalid frequency , chose between: ", ['monthly','quarterly'] )
    fred_md   = pd.read_csv(f"data/current_2023-10_{frequency_dataset}.csv").iloc[1:,:].reset_index(drop=True)
    fred_md= fred_md.rename(columns={"sasdate":'date'})
    fred_md["date"] = pd.to_datetime(fred_md["date"],format="%m/%d/%Y")
    #fred_md = fred_md.iloc[:-2] # last rows had many NaNs
    
    fred_md = fred_md[fred_md['date'] <= max_date]
    
    if min_date is None == False:

        fred_md = fred_md[fred_md['date'] >= min_date]

    prop_nas = (fred_md.isna().sum()/fred_md.shape[0]).reset_index().rename(columns={"index":"col",0:"PropNA"})
 
    #removing all columns that contains NANs
    main_dataframe = fred_md.merge(USREC,on='date',how='left')
    working_dataset = main_dataframe.copy()
    
    cols_drop = prop_nas.query("PropNA > 0")['col'].tolist()

    working_dataset.drop(cols_drop,axis=1,inplace=True)

    
    print("Columns dropped: ", )
    working_dataset=working_dataset[working_dataset['date'].isna() ==False]


    return working_dataset

