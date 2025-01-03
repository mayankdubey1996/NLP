import day1, day2, day3
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
nlp = spacy.load("en_core_web_sm")


def nltk_stemmer(text):
    ps = PorterStemmer()
    return [ps.stem(word) for word in text]

def nltk_lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in text]

def spacy_lemmatizer(text):
    
    doc = nlp(text)
    return [token.lemma_ for token in doc]


if __name__=="__main__":
    dataset_name = "day_1_data.csv"
    df = day1.read_data(dataset_name)
    df["clean_text"] = df["text"].apply(day1.clean_text)

    df["nltk_word_tokenize"] = df["clean_text"].apply(day2.ntlk_word_tokenize)
    df["nltk_sentence_tokenize"] = df["clean_text"].apply(day2.nltk_sent_tokenize)

    df["spacy_word_tokenize"] = df["clean_text"].apply(day2.spacy_word_tokenoizer)
    df["spacy_sentence_tokenize"] = df["clean_text"].apply(day2.spacy_sent_tokenoizer)

    df["nltk_stopwords"] = df["nltk_word_tokenize"].apply(day3.nltk_stopwords)

    df["spacy_stopwords"] = df["clean_text"].apply(day3.spacy_stopwords)

    df["nltk_stemmer"] = df["nltk_stopwords"].apply(nltk_stemmer)
    df["nltk_lemmatization"] = df["nltk_stopwords"].apply(nltk_lemmatizer)

    df["spacy_lemmatization"] = df["clean_text"].apply(spacy_lemmatizer)

    print(df["spacy_lemmatization"])
    print("\n\n\n")

    df["lemmatizer_comparison"] = df["nltk_lemmatization"]==df["spacy_lemmatization"]

    print("Comparing stop words and priting where they are different")
    for index in df[df["lemmatizer_comparison"]==False].index:
        print("Index: ",index)
        print("Nltk: ",df.loc[index,"nltk_lemmatization"])
        print("Spacy: ",df.loc[index,"spacy_lemmatization"])
        print("\n")

    