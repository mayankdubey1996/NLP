import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
import day1
import numpy as np

def ntlk_word_tokenize(text):
    return word_tokenize(text)

def nltk_sent_tokenize(text):
    return sent_tokenize(text)

def spacy_word_tokenoizer(text):
    nlp = spacy.blank("en")
    doc = nlp(text)
    return [token.text for token in doc]

def spacy_sent_tokenoizer(text):
    nlp = spacy.blank("en")
    if "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    doc = nlp(text)
    return [sent.text for sent in doc.sents]

if __name__ == "__main__":
    dataset_name = "day_1_data.csv"
    df = day1.read_data(dataset_name)
    df["clean_text"] = df["text"].apply(day1.clean_text)

    df["nltk_word_tokenize"] = df["clean_text"].apply(ntlk_word_tokenize)
    df["nltk_sentence_tokenize"] = df["clean_text"].apply(nltk_sent_tokenize)

    df["spacy_word_tokenize"] = df["clean_text"].apply(spacy_word_tokenoizer)
    df["spacy_sentence_tokenize"] = df["clean_text"].apply(spacy_sent_tokenoizer)

    df["word_token_comparison"] = df["nltk_word_tokenize"]==df["spacy_word_tokenize"]
    df["sentence_token_comparison"] = df["nltk_sentence_tokenize"]==df["spacy_sentence_tokenize"]


    print("Comparing word tokenization and priting where they are different")
    for index in df[df["word_token_comparison"]==False].index:
        print("Index: ",index)
        print("Nltk: ",df.loc[index,"nltk_word_tokenize"])
        print("Spacy: ",df.loc[index,"spacy_word_tokenize"])
        print("\n")
    
    print("Comparing sentence tokenization and priting where they are different")
    for index in df[df["sentence_token_comparison"]==False].index:
        print("Index: ",index)
        print("Nltk: ",df.loc[index,"nltk_sentence_tokenize"])
        print("Spacy: ",df.loc[index,"spacy_sentence_tokenize"])
        print("\n")
