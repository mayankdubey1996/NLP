import pandas as pd
import re
import nltk
import spacy
import subprocess

class DataPreprocessingPipeline:
    def read_data(self,dataset_name):
        df = pd.read_csv(dataset_name)
        return df
    def count_sentence_len(self,text):
        return len(text.split())
    def clean_text(self,text):
        clean_text=re.sub(r'[^\w\s]|[\d]', '', text)
        return clean_text.lower()
    def ntlk_word_tokenize(self,text):
        return nltk.word_tokenize(text)
    def nltk_sent_tokenize(self,text):
        return nltk.sent_tokenize(text)
    def spacy_word_tokenoizer(self,text):
        nlp = spacy.blank("en")
        doc = nlp(text)
        return [token.text for token in doc]
    def spacy_sent_tokenoizer(self,text):
        nlp = spacy.blank("en")
        if "parser" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

        doc = nlp(text)
        return [sent.text for sent in doc.sents]
    def nltk_stopwords(self,text):
        return [word for word in text if word not in nltk.corpus.stopwords.words("english")]
    def spacy_stopwords(self,text):
        nlp = spacy.blank("en")
        doc = nlp(text)
        return [token.text for token in doc if not token.is_stop]
    def nltk_stemmer(self,text):
        ps = nltk.PorterStemmer()
        return [ps.stem(word) for word in text]
    def nltk_lemmatizer(self,text):
        lemmatizer = nltk.WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in text]
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
        df["len_of_unclean_text"] = df["Text"].apply(self.count_sentence_len)
        df["clean_text"] = df["Text"].apply(self.clean_text)
        df["len_of_clean_text"] = df["clean_text"].apply(self.count_sentence_len)
        df["nltk_word_tokenize"] = df["clean_text"].apply(self.ntlk_word_tokenize)
        df["nltk_sentence_tokenize"] = df["clean_text"].apply(self.nltk_sent_tokenize)
        df["spacy_word_tokenize"] = df["clean_text"].apply(self.spacy_word_tokenoizer)
        df["spacy_sentence_tokenize"] = df["clean_text"].apply(self.spacy_sent_tokenoizer)
        df["nltk_stopwords"] = df["nltk_word_tokenize"].apply(self.nltk_stopwords)
        df["spacy_stopwords"] = df["clean_text"].apply(self.spacy_stopwords)
        df["nltk_stemmer"] = df["nltk_stopwords"].apply(self.nltk_stemmer)
        df["nltk_lemmatization"] = df["nltk_stopwords"].apply(self.nltk_lemmatizer)
        df["spacy_lemmatization"] = df["clean_text"].apply(self.spacy_lemmatizer)
        return df
    
if __name__=="__main__":
    dataset_name = "mini-project-data/BBC News Train.csv"
    pipeline = DataPreprocessingPipeline()
    df = pipeline.data_preprocessing_pipeline(dataset_name)
    print(df)    