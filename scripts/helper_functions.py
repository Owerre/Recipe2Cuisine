# Created by Solomon Owerre.
import nltk
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from nltk.stem import WordNetLemmatizer, PorterStemmer
#############################################################################


def join_tokens(data, text):
    """This function combines all the tokens in one list"""
    token_list = data[text].to_list()
    all_tokens = [tokens for sub_tokens in token_list for tokens in sub_tokens]
    return token_list, all_tokens
#############################################################################


def word_count(tokens):
    """Plot the most frequency words in the corpus"""
    token_count = Counter(tokens)  # count each word
    # count most common 10 words
    top_words = dict(token_count.most_common(n=10))
    freq_plot = pd.Series(top_words, index=None).plot(
        kind='bar', figsize=(12, 6))
    plt.ylabel('Number of occurrences', fontsize=15)
    plt.xlabel('Ingredients', fontsize=15)
    plt.xticks(rotation=70, fontsize=15)
    plt.yticks(fontsize=15)
    plt.suptitle('Most frequent ingredients', fontsize=20)
#############################################################################


def my_tokenizer(text):
    """Text preprocessing, tokenizing all the ingredients"""
    text = text.lower()  # lower case
    text = re.sub("\d+", " ", text)  # remove digits
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    token = [word for word in tokenizer.tokenize(text) if len(word) > 2]
    token = [lemmatizer.lemmatize(x) for x in token]
    token = [s for s in token if s not in stopwords.words('english')]
    return token
#############################################################################


def bag_of_words(data, text):
    """count vectorizer"""
    vectorizer = CountVectorizer(tokenizer=my_tokenizer)
    X = vectorizer.fit_transform(data[text])
    return vectorizer, X
#############################################################################


def tfidf_vectorizer(data, text):
    """Tfidf vectorizer"""
    vectorizer = TfidfVectorizer(tokenizer=my_tokenizer)
    X = vectorizer.fit_transform(data[text])
    return vectorizer, X
#############################################################################


def word2vec_embedding(list_of_tokens):
    """Train word2vec on the corpus"""
    num_features = 300
    min_word_count = 1
    num_workers = 2
    window_size = 6
    subsampling = 1e-3

    model = Word2Vec(list_of_tokens, workers=num_workers,
                     size=num_features, min_count=min_word_count,
                     window=window_size, sample=subsampling)
    return model
#############################################################################


def plot_confusion_matrix(conf_mx, labels=None, title=''):
    """Plot of the confusion matrix"""
    plt.figure(figsize=(20, 10))
    conf_mx = conf_mx / conf_mx.sum(axis=1)  # Normalization
    cm_df = pd.DataFrame(conf_mx, index=[i for i in labels], columns=[
                         i for i in labels])
    sns.heatmap(round(cm_df, 2), annot=True, annot_kws={"size": 14})
    plt.title(title, fontsize=30)


def model_prediction(model, n_training_samples, n_training_labels, n_test_samples, n_test_labels):
    """ Model prediction on the test set.
        Returns test accuracy, f1 score, and confusion matrix
    """
    # Fit the training set
    model.fit(n_training_samples, n_training_labels)

    # Make prediction on the test set
    y_predict = model.predict(n_test_samples)

    # Compute the accuracy of the model
    accuracy = accuracy_score(n_test_labels, y_predict)

    # Compute the accuracy of the model
    f1 = f1_score(n_test_labels, y_predict, average='weighted')

    # Confusion matrix
    conf_mx = confusion_matrix(n_test_labels, y_predict)

    return accuracy, f1, conf_mx
