import  day1, day2
import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy

def nltk_stopwords(text):
    return [word for word in text if word not in stopwords.words("english")]

def spacy_stopwords(text):
    nlp = spacy.blank("en")
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop]

if __name__=="__main__":
    dataset_name = "day_1_data.csv"
    df = day1.read_data(dataset_name)
    df["clean_text"] = df["text"].apply(day1.clean_text)

    df["nltk_word_tokenize"] = df["clean_text"].apply(day2.ntlk_word_tokenize)
    df["nltk_sentence_tokenize"] = df["clean_text"].apply(day2.nltk_sent_tokenize)

    df["spacy_word_tokenize"] = df["clean_text"].apply(day2.spacy_word_tokenoizer)
    df["spacy_sentence_tokenize"] = df["clean_text"].apply(day2.spacy_sent_tokenoizer)

    df["nltk_stopwords"] = df["nltk_word_tokenize"].apply(nltk_stopwords)

    df["spacy_stopwords"] = df["clean_text"].apply(spacy_stopwords)

    df["stopword_comparison"] = df["nltk_stopwords"]==df["spacy_stopwords"]
    
    print("Comparing stop words and priting where they are different")
    for index in df[df["stopword_comparison"]==False].index:
        print("Index: ",index)
        print("Nltk: ",df.loc[index,"nltk_stopwords"])
        print("Spacy: ",df.loc[index,"spacy_stopwords"])
        print("\n")
    

