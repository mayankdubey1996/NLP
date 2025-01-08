from sklearn.feature_extraction.text import TfidfVectorizer

def tf_idf_model(corpus):
    # stop_words="english" removes the common words like "is", "the", "and" etc
    tfidf_vectorizer  = TfidfVectorizer(stop_words="english")
    X = tfidf_vectorizer.fit_transform(corpus)
    
    #  (total document, unique-words)  tf

    return X.toarray (), tfidf_vectorizer 


def tfidf_bigram_model(corpus):
    tfidf_vectorizer  = TfidfVectorizer(ngram_range=(2,2),stop_words="english")

    # ngram_range is a tuple of (min_n, max_n) which means the minimum and maximum number of words in a n-gram.
    #(1, 1): Only unigrams.
    #(2, 2): Only bigrams.
    #(1, 3): Unigrams, bigrams, and trigram

    X = tfidf_vectorizer.fit_transform(corpus)
    return X.toarray(), tfidf_vectorizer

if __name__ =="__main__":
    corpus = ["I am learning NLP and I love NLP" , "Natural Language Processing is fun", "NLP is the future", "The future is here",
              "We can shape the future with NLP", "NLP is the best", "NLP is awesome"]
    
    print("len of corpus:",len(corpus))


    # returns X vector and vectorizer object
    X, tf_idfvectorizer = tf_idf_model(corpus)
    
    # represents the X vector
    print(X)
    # represents the shape of the X vector
    # shape of X vector is (number of documents, number of unique words)
    print(X.shape)
    # represents the number of unique words in the corpus
    print(len(tf_idfvectorizer.get_feature_names_out()))
    # represents unique words in the corpus
    print(tf_idfvectorizer.get_feature_names_out())

    print("Bigram model")

    bigram_X, bigram_vectorizer_tf_idf = tfidf_bigram_model(corpus)
    # represents the X vector
    print(tfidf_bigram_model)
    # represents the shape of the X vector
    print(print(bigram_X.shape))
    # shape of X vector is (number of documents, number of unique words)
    print(bigram_X.shape)
    # represents the number of unique words in the corpus
    print(len(bigram_vectorizer_tf_idf.get_feature_names_out()))
    # represents unique words in the corpus
    print(bigram_vectorizer_tf_idf.get_feature_names_out())


