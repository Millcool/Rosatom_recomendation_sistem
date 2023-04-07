from django.shortcuts import render, redirect
from .forms import ReviewForm
from .models import Review
import pandas as pd
from .functions_for_data import *
from django.http import HttpResponse
from django.views.decorators.http import require_POST
from django.http import FileResponse


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


def index_page(request):
    return render(request, 'index.html')

def upload(request):
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            # perform sentiment analysis on the text to determine rating and positivity

            review.save()
            return redirect('review')
    else:
        form = ReviewForm()
    return render(request, 'review.html', {'form': form})



def review(request):
    #print(f'dfgfdgfdgdfgdfgdgdfgfdgdgdfgdfgfdg{request.method}')
    if request.method == 'POST':
        myfile = request.FILES['myfile']
        name = myfile.name
        print(name)
        if name.endswith('.csv'):
            data = pd.read_csv(myfile)
        elif name.endswith('.xlsx'):
            data = pd.read_excel(myfile)
        else:
            return render(request, 'review.html')

        most_15_not_stem, dictionary_of_most_words, openai_answer = do_all(data)
        print(most_15_not_stem)
        print(openai_answer)
        request.session['items'] = [most_15_not_stem, dictionary_of_most_words, openai_answer]
        make_pdf(most_15_not_stem, dictionary_of_most_words, openai_answer)
        return redirect('upload_success')
        # Обработка загруженного файла
        #return render(request, 'upload_success.html')
    return render(request, 'review.html')

def upload_success(request):
    # Извлекаем список элементов из сессии
    items = request.session.get('items', [])
    context = {'items': items}
    #context = items
    return render(request, 'upload_success.html', context)

def generate_pdf(request):
    items = request.GET.getlist('items')
    most_15_not_stem, dictionary_of_most_words, openai_answer = items[0], items[1], items[2]
    # print(items[0], items[1], items[2])

    # title = "Сводный отчет о проанализированных отзывах на вопрос: \"Какая «большая цель» Росатома может вдохновить Вас на работу и наполнить смыслом ваш ежедневный труд?\""
    # pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
    # # Список наиболее встречающихся слов
    # word_list = most_15_not_stem
    #
    # # Краткий обзор
    # summary = "Краткий обзор: "
    #
    # # Ответ на вопрос
    # answer = openai_answer
    # # Создаем PDF документ
    # packet = io.BytesIO()
    # can = canvas.Canvas(packet, pagesize=letter)
    #
    # # Заголовок
    # can.setFont('DejaVuSans', 14)
    # can.drawString(1 * inch, 10 * inch, title)
    #
    # # Список слов
    # can.setFont('DejaVuSans', 12)
    # can.drawString(1 * inch, 9 * inch, "15 наиболее встречающихся слов в отзывах:")
    # for i, word in enumerate(word_list):
    #     can.drawString(1.2 * inch, (8.8 - i * 0.2) * inch, "- " + word)
    #
    # # Краткий обзор
    # can.setFont('DejaVuSans', 12)
    # can.drawString(1 * inch, 6 * inch, summary)
    # # Ответ на вопрос
    # can.setFont('DejaVuSans', 12)
    # can.drawString(1 * inch, 5.5 * inch, answer)
    #
    # can.save()
    #
    # # Сохраняем PDF документ в файл
    # packet.seek(0)
    response = FileResponse(packet, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="report.pdf"'
    return response

@require_POST
def download_pdf(request):
    data = request.POST.dict()
    pdf_bytes = generate_pdf(data)

    response = HttpResponse(pdf_bytes, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="review.pdf"'

    return response