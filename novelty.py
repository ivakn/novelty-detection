from nltk.corpus import reuters 

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer 
from os.path import expanduser
import numpy as np
from sklearn import svm, metrics ,model_selection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


cachedStopWords = stopwords.words("english")
def clean(sentence):
    sentence = sentence.lower()                 # Converting to lowercase
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence)
    sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence) 
    return sentence

def tokenize(text):
    text=clean(text)
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text));
    words = [word for word in words
                  if word not in cachedStopWords]
    lematized_words =(list(map(lambda token: WordNetLemmatizer().lemmatize(token),
                  words)));
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter(lambda token:
                  p.match(token) and len(token)>=min_length,
         lematized_words));
    return filtered_tokens


def tf_idf(docs):
    tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=3,
                        max_df=0.90, max_features=3000,
                        use_idf=True, sublinear_tf=True,
                        norm='l2');
    tfidf.fit(docs);
    return tfidf;

def feature_values(doc, representer):
    doc_representation = representer.transform([doc])
    features = representer.get_feature_names()
    return np.r_[[doc_representation[0, index]
                 for index in doc_representation.nonzero()[1]]]

def oneClassSvm(X_train,X_test,ground_truth):
    clf=svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
    result=clf.predict(X_test)
    # print(result)
    n_error_train = result[result == -1].size
    n_error_train2 = result[result == 1].size
    result=np.where(result==-1, 0, result) 
    recall=recall_score(result,ground_truth,average="binary")
    accuracy=accuracy_score(result,ground_truth)
    presicion=average_precision_score(result,ground_truth)
    print("OneClassSVM")
    print("Presicion :"+ str(presicion))
    print("Accuracy :"+ str(accuracy))
    print("Recall :"+ str(recall))

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, np.ravel(label))
    predictions = classifier.predict_proba(feature_vector_valid)
    # predict the labels on validation dataset
    return predictions

def max_probability(predictions,eps):
    result=[]
    for p in predictions:
        if max(p)>eps: result.append(1)
        else: result.append(0)
    
    return result

def two_max_probability(predictions,eps):
    result=[]
    for p in predictions:
        p.sort() 
        if p[-1]**2-p[-2]**2<=eps: result.append(0)
        else: result.append(1)
        
    return result
    
def max_prob_classifiers(classifiers,X_train, labels, X_test,ground_truth):
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        probabilities=train_model(clf, X_train, labels, X_test)
        result=max_probability(probabilities,0.9)
        recall=recall_score(result,ground_truth,average="binary")
        accuracy=accuracy_score(result,ground_truth)
        presicion=average_precision_score(result,ground_truth)
        print(clf_name)
        print("Presicion :"+ str(presicion))
        print("Accuracy :"+ str(accuracy))
        print("Recall :"+ str(recall))

def two_max_prob_classifiers(classifiers,X_train, labels, X_test,ground_truth):
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        probabilities=train_model(clf, X_train, labels, X_test)
        result=two_max_probability(probabilities,0.85)
        recall=recall_score(result,ground_truth,average="binary")
        accuracy=accuracy_score(result,ground_truth)
        presicion=average_precision_score(result,ground_truth)
        print(clf_name)
        print("Presicion :"+ str(presicion))
        print("Accuracy :"+ str(accuracy))
        print("Recall :"+ str(recall))
def find_epsilon_max_prob(clf,X_train, labels, X_test,ground_truth):
    fig,  (axs1, axs2, axs3) = plt.subplots(1, 3)
    t = np.arange(0., 1., 0.05)
    recall=[]
    precision=[]
    accuracy=[]
    for ts in t:
        res=train_model(clf, X_train, labels, X_test)
        recall.append(recall_score(max_probability(res,ts),ground_truth,average="binary"))
        precision.append(average_precision_score(max_probability(res,ts),ground_truth))
        accuracy.append(accuracy_score(max_probability(res,ts),ground_truth))
    axs1.plot(t, recall, 'r--')
    axs1.set_xlabel("Epsilon")
    axs1.set_ylabel("Recall")
    axs2.plot(t, precision, 'r--')
    axs2.set_xlabel("Epsilon")
    axs2.set_ylabel("Precision")
    axs3.plot(t, accuracy, 'r--')
    axs3.set_xlabel("Epsilon")
    axs3.set_ylabel("Accuracy")
    plt.show()

def main():

    classifiers = {
    "SVM": svm.SVC(probability=True,kernel="rbf", gamma=0.1),
    "MLPClassifier" :MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=500,criterion='gini'),
    "Logistic Regression": LogisticRegression(solver='lbfgs')}


    train_docs = []
    test_docs = []
    all_docs=[]   
    home = expanduser("~")
    id2cat = dict()
    id2cat2 = dict()

    for line in open(home+'/nltk_data/corpora/reuters/cats.txt','r'):
        fid, _, cats = line.partition(' ')
        if(len(cats.split())==1 and (cats.split()[0]=='earn' or cats.split()[0]=='acq')):
            id2cat[fid] = cats.split()
        if(len(cats.split())==1 and cats.split()[0]=='interest'):
            id2cat2[fid] = cats.split()


    labels=[];
    for doc_id in id2cat:
        if doc_id.startswith("train"):
            train_docs.append(reuters.raw(doc_id))
            labels.append(id2cat[doc_id])
            all_docs.append(reuters.raw(doc_id))
        else:
            test_docs.append(reuters.raw(doc_id))
            all_docs.append(reuters.raw(doc_id))
    for doc_id in id2cat2:
        if doc_id.startswith("test"):
            test_docs.append(reuters.raw(doc_id))
            all_docs.append(reuters.raw(doc_id))
    ground_truth = np.ones(1860, dtype=int)
    ground_truth[-81:] = 0
 
    vectorizer = TfidfVectorizer(tokenizer=tokenize, min_df=3,
                        max_df=0.90, max_features=3000,
                        use_idf=True, sublinear_tf=True,
                        norm='l2')
    vectorizer.fit(all_docs)

    X_train = vectorizer.transform(train_docs)
    X_test  = vectorizer.transform(test_docs)

    oneClassSvm(X_train,X_test,ground_truth)

    # print(train_model(svm.SVC(probability=True,kernel="rbf", gamma=0.1), X_train, labels, X_test,ground_truth,0.85))

    
    # max_prob_classifiers(classifiers,X_train, labels, X_test,ground_truth)
    max_prob_classifiers(classifiers,X_train, labels, X_test,ground_truth)
    find_epsilon_max_prob(svm.SVC(probability=True,kernel="rbf", gamma=0.1),X_train, labels, X_test,ground_truth)


main()


