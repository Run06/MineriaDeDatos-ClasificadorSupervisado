from sklearn.feature_extraction.text import TfidfVectorizer


def dividir_mensajes(df):
    return df

def tfidf(X_train, X_dev, X_test):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_dev_vec = vectorizer.transform(X_dev)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_dev_vec, X_test_vec