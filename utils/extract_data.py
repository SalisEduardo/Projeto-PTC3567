
import pandas as pd 
from fredapi import Fred


DEFAULT_SERIES  = ["USREC", # US Recession
            "GDPC1", "PIECTR", "PRS85006013", "IPB50001SQ", "CQRMTSPL", # Macro
            "T10Y3M", "T10Y2Y", ## Term Spreads
            "BAMLH0A0HYM2", "BAA10Y", "AAAFF",  ## Corporate spreads
            "T10YIE", "T5YIE", ## Inflation spreads
            "VIXCLS", "GVZCLS","OVXCLS", # Volatility Index
            "ATLSBUSRGEP","ATLSBUEGEP",# Business Expectations
            "ATLSBUSRGUP","ATLSBUEGUP",# Business Uncertainty
            ]



def get_FRED_series(series_codes = DEFAULT_SERIES ,APIKEY = "3350621adf83e98b7ae0cf60ee5098a1",return_info=True):
    

    fred = Fred(api_key=APIKEY)

    data = []
    series_infos = []
    for serie in series_codes:
        try:
            info = pd.DataFrame(fred.get_series_info(serie)).T
            series_infos.append(info)
            df = fred.get_series(serie)
            df = df.to_frame()
            df.index = pd.to_datetime(df.index)
            df.columns= [serie]
            data.append(df)
        except Exception as e:
            print(e)
            series_codes.remove(serie)
            print(f"Couldn't get {serie}")
    
    infos = pd.concat(series_infos)

    tickers_dataset = dict(zip(series_codes,data))

    if return_info:
        return({"Infos":infos,
                "Data":tickers_dataset})
    else:
        return(tickers_dataset)
    
def get_data_dictionary(tickers_dataset,infos_dataset,possible_targets = ['USREC']):
    

    # Create a dictionary to store the JSON structure
    infos_dictionary = {}

    # Iterate through each row in the DataFrame
    for index, row in infos_dataset.iterrows():
        # Extract the key from the 'id'
        key = row['id']
        
        # Remove the 'id' from the row data
        row_data = row.drop('id').to_dict()
        
        row_data['ts'] =tickers_dataset[key]

        if key in possible_targets:
            row_data['modeling_variable_type'] = 'Target'
        else:
            row_data['modeling_variable_type'] = 'Attribute'

        # Add the key-value pair to the JSON dictionary
        infos_dictionary[key] = row_data

        return(infos_dictionary)
    
#def get_tickers_frequency():
