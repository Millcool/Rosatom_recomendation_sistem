from django.shortcuts import render, redirect
from .forms import ReviewForm
from .models import Review
from .rating import rating_for_text, predict_pos_neg


def index_page(request):
    return render(request, 'index.html')

def review(request):
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            # perform sentiment analysis on the text to determine rating and positivity
            rating = rating_for_text(text)
            positive = predict_pos_neg(text)
            review = Review(text=text, rating=rating, positive=positive)
            review.save()
            return redirect('review')
    else:
        form = ReviewForm()
    return render(request, 'review.html', {'form': form})