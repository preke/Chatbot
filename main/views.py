# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
import os
import json
import requests
import numpy as np
import re
from transformers import Trainer, TrainingArguments
from transformers import T5Tokenizer, T5ForConditionalGeneration#, Trainer, TrainingArguments

import pdb

from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

np.random.seed(0)


OUTPUT_MAX_LEN=70
OUTPUT_MIN_LEN=1
INPUT_MAX_LENGTH = 128


MODEL_PATH_DICT={}
MODEL_PATH_DICT['model_name']='t5_base_PELD'
MODEL_PATH_DICT['model_path']="/home/disk1/data/shuaiqi/emo_response_gen/model/t5_base/PELD/checkpoint-3906"


emo_cls_tokenizer = AutoTokenizer.from_pretrained("/home/zhiyuan/aaai'23/src/bert_emotion_classification/")  #bert_emotion_classification
emo_cls_model = AutoModelForSequenceClassification.from_pretrained("/home/zhiyuan/aaai'23/src/bert_emotion_classification/", num_labels=6)   #"bert_emotion_classification"
emotion_mapping = {4: "sadness", 2: "joy", 1: "fear", 0: "anger", 5: "surprise", 3: "love"}






# from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
#
# mname = "facebook/blenderbot-400M-distill"
# model = BlenderbotForConditionalGeneration.from_pretrained(mname)
# tokenizer = BlenderbotTokenizer.from_pretrained(mname)

# inputs = tokenizer([UTTERANCE], return_tensors="pt")
# reply_ids = model.generate(**inputs)


def index(request):
    return render(request, 'main/index.html')


@csrf_exempt
def auto_response(request):
    template = loader.get_template('main/index.html')
    post = request.POST['post']

    uttr         = post.split('___')[0]
    _personality = post.split('___')[1]

    personality = person_dict[_personality]

    user_emo = emotionInferenceBERT(uttr)
    response_emo = emotionGenerationBERT(user_emo, uttr, personality)

    # response generation
    generator_dict={}

    # only finetuned on PELD
    # generator_dict['model_name']='t5_base_DailyDialog_PELD' 
    # generator_dict['model_path']="/home/disk1/data/guest01/emo_response_gen/model/t5_base/PELD/checkpoint-3906"
    # only finetuned on DailyDialog
    generator_dict['model_name']='t5_base_DailyDialog'
    generator_dict['model_path']="/home/disk1/data/guest01/emo_response_gen/model/t5_base/DailyDialog/checkpoint-47500"
    # finetuned on DailyDialog and PELD
    # generator_dict['model_name']='t5_base_DailyDialog_PELD'
    # generator_dict['model_path']="/home/disk1/data/guest01/emo_response_gen/model/t5_base/DailyDialog_PELD/checkpoint-2604"

    emo_generator = emotional_response_gen(generator_dict, device = "cuda:1" )

    response_text = emo_generator.response_generate(uttr, 
                                                    emotion = response_emo, 
                                                    personality = _personality, 
                                                    output_max_len = OUTPUT_MAX_LEN, 
                                                    output_min_len = OUTPUT_MIN_LEN, 
                                                    input_max_length = INPUT_MAX_LENGTH)

    res_list = {
        'user_emo': 'It seems your current emotion is ' + user_emo + '.',
        'response': response_text,
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






# ******************************************************************


def emotionInferenceBERT(input_sentence):
    inputs = emo_cls_tokenizer(input_sentence, return_tensors = "pt")
    outputs = emo_cls_model(**inputs)
    label_idx = torch.argmax(outputs[0]).item()
    return emotion_mapping[label_idx]

def emotionGenerationBERT(user_emo, user_post, personality):
    if personality == 'A':
        response_emo_prob = np.array([i/sum(A_dict[user_emo]) for i in A_dict[user_emo]])
        index = np.random.choice([0, 1, 2, 3, 4, 5], p = response_emo_prob.ravel())
        response_emo = emo_dict[index]
    elif personality == 'C':
        response_emo_prob = np.array([i/sum(C_dict[user_emo]) for i in C_dict[user_emo]])
        index = np.random.choice([0, 1, 2, 3, 4, 5], p = response_emo_prob.ravel())
        response_emo = emo_dict[index]
    elif personality == 'E':
        response_emo_prob = np.array([i/sum(E_dict[user_emo]) for i in E_dict[user_emo]])
        index = np.random.choice([0, 1, 2, 3, 4, 5], p = response_emo_prob.ravel())
        response_emo = emo_dict[index]
    elif personality == 'O':
        response_emo_prob = np.array([i/sum(O_dict[user_emo]) for i in O_dict[user_emo]])
        index = np.random.choice([0, 1, 2, 3, 4, 5], p = response_emo_prob.ravel())
        response_emo = emo_dict[index]
    elif personality == 'N':
        response_emo_prob = np.array([i/sum(N_dict[user_emo]) for i in N_dict[user_emo]])
        index = np.random.choice([0, 1, 2, 3, 4, 5], p = response_emo_prob.ravel())
        response_emo = emo_dict[index]
    return response_emo


person_dict = {
    "Sympathetic"  : "A",
    "Organized"    : "C",
    "Extroverted"  : "E",
    "Insightful"   : "O",
    "Sensitive"    : "N"
}

emo_dict = {
    0: 'joy',
    1: 'surprise',
    2: 'anger',
    3: 'sadness',
    4: 'fear',
    5: 'disgust' 
}


O_dict = {
    "joy": [0.39, 0.05, 0.06, 0.05, 0.06, 0.01], 
    "surprise": [0.15, 0.24, 0.11, 0.05, 0.02, 0.08], 
    "anger": [0.12, 0.08, 0.38, 0.09, 0.04, 0.02], 
    "sadness": [0.17, 0.08, 0.09, 0.25, 0.05, 0.02], 
    "fear": [0.19, 0.02, 0.21, 0.12, 0.21, 0.00], 
    "disgust": [0.07, 0.13, 0.07, 0.13, 0.07, 0.20]
}


C_dict = {
    "joy": [0.37, 0.08, 0.07, 0.06, 0.06, 0.00], 
    "surprise": [0.15, 0.27, 0.10, 0.04, 0.01, 0.08], 
    "anger": [0.11, 0.07, 0.44, 0.05, 0.07, 0.01], 
    "sadness": [0.09, 0.08, 0.17, 0.25, 0.09, 0.03], 
    "fear": [0.11, 0.03, 0.20, 0.09, 0.35, 0.01], 
    "disgust": [0.11, 0.15, 0.19, 0.04, 0.04, 0.22]
}


A_dict = {
    "joy": [0.29, 0.06, 0.0, 0.03, 0.09, 0.02], 
    "surprise": [0.11, 0.19, 0.09, 0.06, 0.03, 0.08], 
    "anger": [0.07, 0.05, 0.41, 0.09, 0.07, 0.02], 
    "sadness": [0.09, 0.05, 0.18, 0.23, 0.09, 0.03], 
    "fear": [0.15, 0.05, 0.15, 0.06, 0.25, 0.01], 
    "disgust": [0.10, 0.05, 0.15, 0.0, 0.0, 0.25]
}


E_dict = {
    "joy": [0.32, 0.10, 0.07, 0.05, 0.07, 0.02], 
    "surprise": [0.15, 0.30, 0.14, 0.08, 0.06, 0.03], 
    "anger": [0.07, 0.05, 0.59, 0.06, 0.04, 0.02], 
    "sadness": [0.08, 0.10, 0.09, 0.30, 0.12, 0.03], 
    "fear": [0.22, 0.05, 0.13, 0.10, 0.28, 0.01], 
    "disgust": [0.00, 0.24, 0.18, 0.12, 0.00, 0.12]
}


N_dict = {
    "joy": [0.27, 0.06, 0.07, 0.08, 0.09, 0.00], 
    "surprise": [0.24, 0.24, 0.14, 0.05, 0.02, 0.00], 
    "anger": [0.09, 0.04, 0.42, 0.07, 0.07, 0.02], 
    "sadness": [0.09, 0.05, 0.10, 0.32, 0.05, 0.01], 
    "fear": [0.10, 0.03, 0.17, 0.06, 0.36, 0.00], 
    "disgust": [0.13, 0.07, 0.20, 0.0, 0.0, 0.20]
}


class emotional_response_gen(object):

    def __init__(
            self,
            generator_dict, 
            device = "cuda"   
    ):
        self.generator_dict = generator_dict
        #self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.torch_device = device if torch.cuda.is_available() else 'cpu'
        self.generator_name = generator_dict['model_name'] 
        self.generator_model_path = generator_dict['model_path']
        #self.input_text = input_text
        #self.section_name = section_name
        #self.summarizer_name = summarizer_name
        #self.output_len = output_len
        #self.input_truncate_len = input_len
        if "t5_base" in self.generator_name:
            #pdb.set_trace()
            self.tokenizer = T5Tokenizer.from_pretrained(self.generator_model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(self.generator_model_path).to(self.torch_device)

    
    def preprocess_input_text(self, input_str):
        
        input_str = str(input_str).strip().lower()
        input_str = input_str.replace('\\n',' ').replace('\n',' ').replace('\r',' ').replace('——',' ').replace('——',' ').replace('__',' ').replace('__',' ').replace('........','.').replace('....','.').replace('....','.').replace('..','.').replace('..','.').replace('..','.').replace('. . . . . . . . ','. ').replace('. . . . ','. ').replace('. . . . ','. ').replace('. . ','. ').replace('. . ','. ')
        input_str = input_str.encode('unicode_escape').decode('ascii')
        input_str = re.sub(r'\\u[0-9a-z]{4}', ' ', input_str)
        input_str = input_str.replace('\\ud835',' ').replace('    ',' ').replace('  ',' ').replace('  ',' ')
        return input_str
        
    def add_prefix(self, input_str,emotion,personality):
        prefix_str = "Dialog: " + "Emotion: " + emotion + ". Personality: " + personality + ". "
        prefix_added_str = prefix_str + input_str
        
        return prefix_added_str


    def response_generate(self, input_text, emotion = "joy", personality = "extraversion", output_max_len=70, output_min_len=1, input_max_length = 128):
        #output_text = ""
        cleaned_input_text = self.preprocess_input_text(input_text)
        prefix_added_text = self.add_prefix(cleaned_input_text,emotion,personality)

        test_batch = [prefix_added_text]

        #torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        inputs = self.tokenizer(test_batch,truncation=True, padding=True, max_length=input_max_length, return_tensors='pt').to(self.torch_device)

        predictions = self.model.generate(**inputs,max_length=output_max_len,min_length=output_min_len,num_beams=5,length_penalty=2.0,no_repeat_ngram_size=3)
        predictions = self.tokenizer.batch_decode(predictions)

        output_response_list = []
        for prediction in predictions:
            #cleaned_prediction = prediction.strip().replace('\n',' ').replace('\r',' ').replace('  ',' ').replace('  ',' ')
            cleaned_prediction = prediction.strip().replace('\n',' ').replace('\r',' ').replace('<pad>','').replace('</s> ','').replace('</s>','').replace('<s>','').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
            output_response_list.append(cleaned_prediction)

        response_text = output_response_list[0]
        return response_text







