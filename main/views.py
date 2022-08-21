# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
import os
import json
import requests


# from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
#
# mname = "facebook/blenderbot-400M-distill"
# model = BlenderbotForConditionalGeneration.from_pretrained(mname)
# tokenizer = BlenderbotTokenizer.from_pretrained(mname)

def index(request):
    return render(request, 'main/index.html')



@csrf_exempt
def auto_response(request):
    template = loader.get_template('main/index.html')
    post = request.POST['post']

    uttr         = post.split('___')[0]
    personlality = post.split('___')[1]

    # inputs = tokenizer([UTTERANCE], return_tensors="pt")
    # reply_ids = model.generate(**inputs)
    
    '''
    addr = '127.0.0.1'
    port = '8080'
    emotion = 'joy'
    url = 'http://%s:%s/cakechat_api/v1/actions/get_response' % (addr, port)
    body = {'context': [post], 'emotion': emotion}
    response = requests.post(url, json=body)
    print(response.json())
    '''

    user_emo = 'Joy'
    response_emo = 'Anger'

    res_list = {
        'user_emo': 'It seems your current emotion is ' + user_emo + '.',
        'response': uttr,
        'response_emo': 'So, I response you with in ' + response_emo+ ':',
    }

    res = json.dumps(res_list)



    # response = UTTERANCE #tokenizer.batch_decode(reply_ids)

    return HttpResponse(res)


@csrf_exempt
def response_api(request):
    template = loader.get_template('main/index.html')
    post = request.POST['info']
    print(post)
    addr = '127.0.0.1'
    port = '8080'
    emotion = 'joy'
    url = 'http://%s:%s/cakechat_api/v1/actions/get_response' % (addr, port)
    body = {'context': [post], 'emotion': emotion}

    response = requests.post(url, json=body)
    print(response.json())
    # print response.json()['response']

    return HttpResponse(response.json()['response'])


def emotionGenerationBERT(user_emo, user_post):
    return response_emo