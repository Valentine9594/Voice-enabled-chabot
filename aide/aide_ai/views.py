from django.shortcuts import render
from . import aide
from django.template import RequestContext

# Create your views here.
def index(request):
    return render(request,"index.html")

def Voice(request):
    if request.method == 'GET':
        quest=str(request.GET.get('Query_text'))
        lang=str(request.GET.get('Language'))

        ans = aide.chat(quest,lang)
        return render(request,'index.html', {'Ans':ans,'Lang':lang})
