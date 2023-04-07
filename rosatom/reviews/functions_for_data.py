from .open_ai_key import *
import openai
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter
import re
import string

import spacy
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
snowball = SnowballStemmer(language="russian")
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileWriter, PdfFileReader
from io import BytesIO
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from PyPDF2 import PdfFileReader, PdfMerger


from wordcloud import WordCloud
from PIL import Image

porter = PorterStemmer()
wnl = WordNetLemmatizer()
nlp = spacy.load('ru_core_news_lg')
stopwords=nlp.Defaults.stop_words


def n_most_common_words(df, n):
    # Вытаскиваем 15 слов
    words = []
    for w in df['review'][:1999]:
        words.extend(w.split())

    word_count = len(words)
    #print(f'Количество слов в списке: {word_count}')

    # нахождение 15 самых повторяющихся слов
    word_counter = Counter(words)
    most_common = word_counter.most_common(15)
    #print(f'15 самых повторяющихся слов: {most_common}')
    most_15_not_stem = [w[0] for w in most_common]
    return most_15_not_stem

def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis("off")

def preprocessing(line, snowball= snowball, token=porter, ):
    print(line)
    line = str(line)
    line = re.sub(r'\d+', '', line)
    # Преобразование строки к нижнему регистру и удаление знаков пунктуации
    line = re.sub(r"[{}]".format(string.punctuation), " ", line.lower())
    # Удаление повторяющихся переводов строк и преобразование остальных в пробелы
    line = re.sub(r'«»"\'', '', line)
    line = re.sub(r'«', '', line)
    line = re.sub(r'»', '', line)
    line = re.sub(r"\n\n+", " ", line).replace('\n', ' ')
    line = re.sub(r"\s+", " ", line)
    # Получение списка токенов с помощью spacy
    doc = nlp(line)
    # Список стоп-слов, преобразованный во множество для быстрого поиска
    stop_words_set = set(stopwords)
    # Преобразование каждой леммы токена с помощью стемминга и проверка на наличие в списке стоп-слов
    line = ' '.join([(token.lemma_) for token in doc if token.text not in stop_words_set])
    return line


def dict_of_words_cluster(reviews, lst, n_of_words):
    words = []
    unique = set(lst)
    dictionary_of_most_words = {}
    for i in range(len(unique)):
        dictionary_of_most_words[i] = []

    dictionary = {}
    for i in range(len(unique)):
        dictionary[i] = []

    for i in range(len(unique)):
        for ind, item in enumerate(lst):
            if item == i:
                dictionary[i].append(ind)

    for key in dictionary:
        for ind, review in enumerate(reviews):
            if ind in dictionary[key]:
                words.extend(review.split())

        # подсчет количества слов в списке
        word_count = len(words)
        #print(f'Количество слов в списке: {word_count}')
        # print(words)
        # нахождение 15 самых повторяющихся слов
        word_counter = Counter(words)
        most_common = word_counter.most_common(n_of_words)
        #print(f'15 самых повторяющихся слов: {most_common}')
        most_15_not_stem = [w[0] for w in most_common]
        dictionary_of_most_words[key] = most_15_not_stem
        words = []
    return dictionary_of_most_words


def do_all(data):
    data = data.dropna()
    openai.api_key = OPENAI_KEY
    engine = "text-davinci-003"
    new_df = []
    for review in data['Какая «большая цель» Росатома может вдохновить Вас на работу и наполнить смыслом ваш ежедневный труд?']:
        review = preprocessing(review)
        new_df.append(review)
    df = pd.DataFrame(new_df, columns=['review'])

    most_15_not_stem = n_most_common_words(df, 15)
    prompt = f"Ответь на вопрос Какая «большая цель» Росатома может вдохновить Вас на работу и наполнить смыслом ваш ежедневный труд? в четырех - пяти предложениях . применяя слова {most_15_not_stem}"

    # Модель
    completion = openai.Completion.create(engine=engine,
                                          prompt=prompt,
                                          temperature=0.5,
                                          max_tokens=1000)
    s = ''
    for w in most_15_not_stem:
        s += w
        s += ' '
###################################################################

    tfidf = TfidfVectorizer(
        preprocessor=preprocessing
    )
    tfidf_data = (
        tfidf
            .fit_transform(df['review'])
            .toarray()
    )

    clusterNum = 15
    k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
    k_means.fit(tfidf_data)
    labels = k_means.labels_
    lst = list(labels)
    # wordcloud = WordCloud(width=2000,
    #                       height=1600,
    #                       random_state=1,
    #                       background_color='white',
    #                       margin=6,
    #                       colormap='plasma',
    #                       collocations=False
    #                       ).generate(s)
    # wordcloud.to_file('loud_simple.png')
    #
    # pca = PCA(n_components=2)
    # pca_decomp_1 = pca.fit_transform(tfidf_data).astype('float16')
    # X = pca_decomp_1
    # y = lst
    # data1 = pd.DataFrame(X)
    # data2 = pd.DataFrame(y, columns=['label'])
    # da = pd.concat([data1, data2], axis=1)
    # #
    # fig = plt.figure()
    # #
    # fig.set_size_inches(16, 10)
    # #
    # sns.scatterplot(x=0,
    #                 y=1,
    #                 hue='label',
    #                 edgecolor="k",
    #                 palette=["#FF5533", "#00B050", "#FFB050", "#00A05F"],
    #                 data=da)
    #
    # #plt.show()
    # fig.savefig('my_plot.png')

    dictionary_of_most_words = dict_of_words_cluster(df['review'], lst, 15)
    return most_15_not_stem, dictionary_of_most_words, completion.choices[0]['text']

def make_pdf(most_15_not_stem, dictionary_of_most_words, openai_answer):
    # Заголовок документа
    title = "Сводный отчет о проанализированных отзывах на вопрос: \"Какая «большая цель» Росатома может вдохновить Вас на работу и наполнить смыслом ваш ежедневный труд?\""
    pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
    # Список наиболее встречающихся слов
    word_list = most_15_not_stem

    # Краткий обзор
    summary = "Краткий обзор: "

    # Ответ на вопрос
    answer = openai_answer
    # Создаем PDF документ
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)

    # Заголовок
    can.setFont('DejaVuSans', 14)
    can.drawString(1 * inch, 10 * inch, title)

    # Список слов
    can.setFont('DejaVuSans', 12)
    can.drawString(1 * inch, 9 * inch, "15 наиболее встречающихся слов в отзывах:")
    for i, word in enumerate(word_list):
        can.drawString(1.2 * inch, (8.8 - i * 0.2) * inch, "- " + word)

    # Краткий обзор
    can.setFont('DejaVuSans', 12)
    can.drawString(1 * inch, 6 * inch, summary)
    # Ответ на вопрос
    can.setFont('DejaVuSans', 12)
    can.drawString(1 * inch, 5.5 * inch, answer)

    can.save()

    # Сохраняем PDF документ в файл
    packet.seek(0)
    new_pdf = PdfMerger()
    new_pdf.add_metadata({'Title': title})
    new_pdf.append(packet)
    with open('report.pdf', 'wb') as output:
        new_pdf.write(output)