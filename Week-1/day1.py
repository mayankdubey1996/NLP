import pandas as pd
import re

def read_data(dataset_name):
    df = pd.read_csv(dataset_name)
    return df

def count_sentence_len(text):
    return len(text.split())

def clean_text(text):
    return re.sub(r'[^\w\s]|[\d]', '', text)

if __name__ =="__main__":
    dataset_name = "day_1_data.csv"
    df = read_data(dataset_name)

    df["len_of_unclean_text"] = df["text"].apply(count_sentence_len)
    df["clean_text"] = df["text"].apply(clean_text)

    df["len_of_clean_text"] = df["clean_text"].apply(count_sentence_len)

    print(df[['len_of_unclean_text','len_of_clean_text']])


    