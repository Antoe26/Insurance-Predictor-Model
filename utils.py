import numpy as np

# function to convert age and income range to midpoint
def convert_age_to_midpoint(value):
    value=str(value).strip().lower()
    if '-' in value:
        try:
            low,high=map(int,value.replace('-','').split('-'))
            return (low + high)/2
        except:
            return np.nan
    elif '60+' in value:
        return 65
    else:
        return np.nan
        
def convert_income_to_midpoint(value):
    value=str(value).lower().strip()
    if '-' in value:
        try:
            low,high=map(int, value.replace('-','').split('-'))
            return (low + high)/2
        except:
            return np.nan
    elif 'less than' in value:
        try:
            number=int(value.replace('less than','').replace('-','').split(''))
            return number/2
        except:
            return np.nan
    else:
        return np.nan