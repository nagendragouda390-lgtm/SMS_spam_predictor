import pandas as pd

print("Reading first csv...")
df1 = pd.read_csv("/storage/8006-15FE/NLP_spam_predictor/SMSSpamCollection",header=None,names=["label","message"],sep="\t") 
# Seperator - space

print("Reading second csv...")
df2 = pd.read_csv("/storage/8006-15FE/NLP_spam_predictor/sms_spam.csv")

print("Concating two dataframes...")
df = pd.concat([df1,df2],ignore_index=True) # adding to dataframe to single df


print("Dropping duplicates...")
df = df.drop_duplicates().reset_index(drop=True) # Dropping duplicates

df.to_csv("/storage/8006-15FE/NLP_spam_predictor/concated.csv",index=False) # uploding csv

print(f"Concated data uploded successfully...")



