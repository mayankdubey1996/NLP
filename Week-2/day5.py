from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def tfidf_vectorization(documents):
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    X = tfidf_vectorizer.fit_transform(documents)
    return X.toarray(), tfidf_vectorizer

def top_n_keywords_by_doc(df,doc_n, top_n):
    return df.iloc[doc_n].sort_values(ascending=False).head(top_n)
    


if __name__ == "__main__":

    documents = [
    "Machine learning is fascinating.",
    "Machine learning and deep learning are subsets of AI.",
    "AI and machine learning are transforming industries."
    ]

    X, tfidf_vectorizer = tfidf_vectorization(documents)

    print("TF-IDF vectorizer:\n",X)
    print(tfidf_vectorizer.get_feature_names_out())

    socred_df = pd.DataFrame(X, columns=tfidf_vectorizer.get_feature_names_out())

    for i in range(len(documents)):
        print("##Document:", documents[i])
        print("Top 2 key words\n",top_n_keywords_by_doc(socred_df,i, 2))

   