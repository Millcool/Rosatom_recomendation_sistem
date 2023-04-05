import re
import string
import pandas as pd
import spacy
import nltk

from sklearn.decomposition import PCA
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier







def rating_for_text(text):
    return 4

def predict_pos_neg(text):
    """
    Some predict of catboost
    :return:
    """
    return True