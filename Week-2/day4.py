import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def read_data(dataset_name):
    df = pd.read_csv(dataset_name)
    return df

def get_row(df, index):
    return df.iloc[index].values.reshape(1,-1)

def cosine_similarity_matrix(X, Y):
    cosine_sim = cosine_similarity(X,Y)
    return cosine_sim

if __name__ == "__main__":

    count_path = "E:/NLP/datasets/mini-project-1-data/week2_day_3count_vectorizer.csv"
    tfidf_path = "E:/NLP/datasets/mini-project-1-data/week_2_day3_tfidf_vectorizer.csv"

    count_df = read_data(count_path)
    tdidf_df = read_data(tfidf_path)

    C_seven_row = get_row(count_df, index=7)
    C_eight_row = get_row(count_df, index=8)

    print("Count Vectorizer vector",cosine_similarity_matrix(C_seven_row, C_eight_row))

    TF_seven_row = get_row(tdidf_df, index=7)
    TF_eight_row = get_row(tdidf_df, index=8)

    print("TF-IDF vector:",cosine_similarity_matrix(TF_seven_row, TF_eight_row))


    

