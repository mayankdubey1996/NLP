import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


class KeywordExtraction:
    def __init__(self):
        pass

    def read_data(self,dataset_name):
        df = pd.read_csv(dataset_name)
        return df

    def clean_text(self,text):
        clean_text=re.sub(r'[^\w\s]|[\d]', '', text)
        return clean_text.lower()

    def tfidf_vectorization(self,documents):
        tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        X = tfidf_vectorizer.fit_transform(documents)
        return X.toarray(), tfidf_vectorizer

    def top_n_keywords_by_doc(self,df, top_n):
        return df.sort_values(ascending=False).head(top_n)

if __name__ == "__main__":
    dataset_path = "E:/NLP/datasets/min-project-2-data/wiki_movie_plots_deduped.csv"

    kc = KeywordExtraction()
    
    df = kc.read_data(dataset_path)
    print("Data read successfully...")
    print("Data shape:",df.shape)

    df["clean_text"] = df["Plot"].apply(kc.clean_text)
    print("Text cleaned successfully...")

    X, tfidf_vectorizer = kc.tfidf_vectorization(df["clean_text"])
    print("TF-IDF vectorization done successfully...")

    socred_df = pd.DataFrame(X, columns=tfidf_vectorizer.get_feature_names_out())

    movie_plot_keywords = defaultdict(list)

    for i in range(len(df)):
        plot = df["Plot"].iloc[i]
        title = df["Title"].iloc[i]

        words_score = kc.top_n_keywords_by_doc(socred_df.iloc[i], 10)

        movie_plot_keywords["title"].append(title)
        movie_plot_keywords["plot"].append(plot)


        for j in range(10):
            movie_plot_keywords["keywords_"+str(j)].append(words_score.index[j])
            movie_plot_keywords["score_"+str(j)].append(words_score.values[j])

    pd.DataFrame(movie_plot_keywords)
    print("Keywords extracted successfully...")

    print("Saving the extracted keywords to a csv file...")
    pd.DataFrame(movie_plot_keywords).to_csv("E:/NLP/datasets/min-project-2-data/movie_plot_keywords.csv", index=False)