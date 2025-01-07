from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words_model(corpus):

    # stop_words="english" removes the common words like "is", "the", "and" etc
    vectorizer = CountVectorizer(stop_words="english")
    
    X = vectorizer.fit_transform(corpus)
    return X.toarray(), vectorizer

def bigram_model(corpus):
    vectorizer = CountVectorizer(ngram_range=(2,2), stop_words="english")

    # ngram_range is a tuple of (min_n, max_n) which means the minimum and maximum number of words in a n-gram.
    #(1, 1): Only unigrams.
    #(2, 2): Only bigrams.
    #(1, 3): Unigrams, bigrams, and trigram

    X = vectorizer.fit_transform(corpus)
    return X.toarray(), vectorizer



if __name__ =="__main__":
    corpus = ["I am learning NLP and I love NLP" , "Natural Language Processing is fun", "NLP is the future", "The future is here",
              "We can shape the future with NLP", "NLP is the best", "NLP is awesome"]
    
    print("len of corpus:",len(corpus))

    # returns X vector and vectorizer object
    X, vectorizer = bag_of_words_model(corpus)
    
    # represents the X vector
    print(X)
    # represents the shape of the X vector
    # shape of X vector is (number of documents, number of unique words)
    print(X.shape)
    # represents the number of unique words in the corpus
    print(len(vectorizer.get_feature_names_out()))
    # represents unique words in the corpus
    print(vectorizer.get_feature_names_out())

    print("Bigram model")

    bigram_X, bigram_vectorizer = bigram_model(corpus)
    # represents the X vector
    print(bigram_X)
    # represents the shape of the X vector
    # shape of X vector is (number of documents, number of unique words)
    print(bigram_X.shape)
    # represents the number of unique words in the corpus
    print(len(bigram_vectorizer.get_feature_names_out()))
    # represents unique words in the corpus
    print(bigram_vectorizer.get_feature_names_out())


