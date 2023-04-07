from django.shortcuts import render, redirect
from .forms import ReviewForm
from .models import Review
import pandas as pd
from .functions_for_data import *
from django.http import HttpResponse
from django.views.decorators.http import require_POST

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

        most_15_not_stem, dictionary_of_most_words, openai_answer = do_all(data)
        print(most_15_not_stem)
        print(openai_answer)
        request.session['items'] = most_15_not_stem, dictionary_of_most_words, openai_answer
        #make_pdf(most_15_not_stem, dictionary_of_most_words, openai_answer)
        return redirect('upload_success')
        # Обработка загруженного файла
        #return render(request, 'upload_success.html')
    return render(request, 'review.html')

def upload_success(request):
    # Извлекаем список элементов из сессии
    items = request.session.get('items', [])
    context = {'items': items}
    return render(request, 'upload_success.html', context)

def generate_pdf(request):
    items = request.GET.get('items', '')
    print(items[0], items[1], items[2])
    make_pdf(items[0], items[1], items[2])


    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="report.pdf"'

    # Create the PDF object, using the response object as its "file."
    # p = canvas.Canvas(response)
    #
    # # Draw things on the PDF. Here's where the PDF generation happens.
    # # See the ReportLab documentation for the full list of functionality.
    # p.drawString(100, 750, "Hello world.")
    #
    # # Close the PDF object cleanly, and we're done.
    # p.showPage()
    # p.save()
    return response

@require_POST
def download_pdf(request):
    data = request.POST.dict()
    pdf_bytes = generate_pdf(data)

    response = HttpResponse(pdf_bytes, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="review.pdf"'

    return response