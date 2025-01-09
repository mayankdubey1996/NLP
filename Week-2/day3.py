import pandas as pd
import re
import spacy
import subprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



class DataPreprocessingPipeline:
    def read_data(self,dataset_name):
        df = pd.read_csv(dataset_name)
        return df

    def clean_text(self,text):
        clean_text=re.sub(r'[^\w\s]|[\d]', '', text)
        return clean_text.lower()
    

    def spacy_word_tokenoizer(self,text):
        nlp = spacy.blank("en")
        doc = nlp(text)
        return [token.text for token in doc]
    
    
    def spacy_lemmatizer(self,text):
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return [token.lemma_ for token in doc]
    
    def data_preprocessing_pipeline(self,dataset_name):
        df = self.read_data(dataset_name)
        print("Data read successfully")
        df["clean_text"] = df["Text"].apply(self.clean_text)
        print("Text cleaned successfully")
        df["spacy_lemmatization"] = df["clean_text"].apply(self.spacy_lemmatizer)
        print("Spacy Lemmatization done successfully")
        return df

class WordVectorization:
    def __init__(self):
        pass

    def count_vectorizer(self,corpus):
        count_vectorizer = CountVectorizer()
        X = count_vectorizer.fit_transform(corpus)
        return X.toarray(), count_vectorizer

    def tf_idf_model(self,corpus):
        tfidf_vectorizer  = TfidfVectorizer(stop_words="english")
        X = tfidf_vectorizer.fit_transform(corpus)
        return X.toarray (), tfidf_vectorizer 

    def word_vectorization_pipeline(self,corpus):
        X_count_vectorizer, count_vectorizer = self.count_vectorizer(corpus)
        X_tfidf_vectorizer, tfidf_vectorizer = self.tf_idf_model(corpus)
        return X_count_vectorizer, count_vectorizer, X_tfidf_vectorizer, tfidf_vectorizer
    
if __name__=="__main__":
    
    """
    1. lets copy the mini project code from week 1
    2. I will be only be using spacy (not nltk) for tokenization and lemmatization
    3. Stop words can be removed by scikit learn's Countvectorizer, TfidfVectorizer
    """

    dataset_name = "E:/NLP/datasets/mini-project-1-data/BBC News Train.csv"
    pipeline = DataPreprocessingPipeline()
    df = pipeline.data_preprocessing_pipeline(dataset_name)

    print("Spacy Lemmatization")
    print(df)    

    corpus = df["spacy_lemmatization"].tolist()
    vectorization = WordVectorization()
    X_count_vectorizer, count_vectorizer, X_tfidf_vectorizer, tfidf_vectorizer = vectorization.word_vectorization_pipeline(corpus)

    print("Count X Vector")
    print(X_count_vectorizer)
    print("Count Vectorizer feature names")
    print(count_vectorizer.get_feature_names_out())

    print("Tfidf X Vector")
    print(X_tfidf_vectorizer)
    print("Tfidf Vectorizer feature names")
    print(tfidf_vectorizer.get_feature_names_out())