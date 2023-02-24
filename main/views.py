# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
import datetime
import requests
import random
import json

# online usage
from transformers import Trainer, TrainingArguments
from transformers import T5Tokenizer, T5ForConditionalGeneration

# import pdb
#
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import argparse

from openprompt import PromptDataLoader
from openprompt.plms import load_plm
from openprompt.data_utils import InputExample
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt import PromptForClassification

# from transformers.tokenization_utils import PreTrainedTokenizer
from yacs.config import CfgNode
from openprompt.data_utils import InputFeatures
import re
from openprompt import Verbalizer
from typing import *
import os
import torch.nn as nn
import torch.nn.functional as F
from openprompt.utils.logging import logger
from openprompt.prompts.manual_template import ManualTemplate
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.prompts import ManualTemplate
from _verbalizer import ManualVerbalizer, KnowledgeableVerbalizer

np.random.seed(0)

device = torch.device('cuda:0')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-base")

device = 0
use_cuda = True

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

mname = "facebook/blenderbot-400M-distill"
response_gen_model = BlenderbotForConditionalGeneration.from_pretrained(mname)
response_gen_tokenizer = BlenderbotTokenizer.from_pretrained(mname)
response_gen_model.eval()
from transformers import pipeline

emo_classifier = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion',
                          return_all_scores=True)

emotion_mapping_2 = {
    'sadness': 'sad',
    'joy': 'happy',
    'fear': 'fearful',
    'anger': 'angry',
    'surprise': 'surprised',
    'love': 'disgusted',
    'neutral': 'neutral'
}

utters = []
file_name = ''


def index(request):
    time_str = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    request.session['time'] = time_str
    request.session['context'] = []
    request.session['rate'] = 0
    return render(request, 'main/index.html')


def get_emo(uttr):
    prediction = emo_classifier(uttr)
    max_score = 0
    label = 'neutral'
    for p in prediction:
        if p['score'] > max_score:
            label = p['label']
            max_score = p['score']

    if max_score < 0.98:
        label = 'neutral'
    #     if label == 'anger' and max_score < 0.98:
    #         label = 'neutral'
    print(max_score, label)
    return label


def get_intent(uttr):
    request_for_understanding = ['i am', 'feel', 'i\'m']
    seek_for_solution = ['can', 'should']
    end_conversation = ['bye', 'thank', 'thanks']

    intent = 'Free Chatting'
    for phrase in request_for_understanding:
        if phrase in uttr.lower():
            intent = 'Request For Understanding'
            return intent
    for phrase in seek_for_solution:
        if phrase in uttr.lower():
            intent = 'Seek For Solution'
            return intent
    for phrase in end_conversation:
        if phrase in uttr.lower():
            intent = 'End Conversation'
            return intent
    print(uttr, intent)
    return intent


# # offline test
# @csrf_exempt
# def auto_response(request):
#     print('Session:', request.session['time'])
#     template = loader.get_template('main/index.html')
#     post = request.POST['post']

#     uttr         = post.split('_')[0]
#     _personality = post.split('_')[1]
#     rate         = post.split('_')[2]


#     file_name = 'records/' + request.session['time']+ '.txt'
#     try:
#         f = open(file_name, 'r')
#         f.close()
#     except:
#         with open(file_name, 'w') as f:
#             f.write(_personality + '\n')

#     request.session['context'].append(uttr)


#     print(request.session['context'])
#     context_ = ' '.join(request.session['context'])

#     response_text = 'abc... '
#     user_emo = 'anger'


#     personality = person_dict[_personality]
#     # response_emo = emotionGenerationBERT(user_emo, uttr, personality)


#     # user_emo = emotion_mapping[random.randint(0, 6)]


#     user_personality = [0,0,0,0,0] #get_personality(context_)
#     user_intent = get_intent(uttr)
#     # context.append(response_text)

#     res_list = {
#         'user_emo': emotion_mapping_2[user_emo],
#         'response': response_text, #'[' + response_emo + ']' +response_text,
#         'user_personality': user_personality,
#         'user_intent': user_intent,
#         'response_emo': 'So'#, I response you with in ' + response_emo + ':',
#     }

#     res = json.dumps(res_list)

#     # response = UTTERANCE #tokenizer.batch_decode(reply_ids)


#     with open(file_name, 'a') as f:
#         f.write(uttr + '\t' + response_text + '\t' + str(request.session['rate']) + '\n')

#     request.session['rate'] = rate
#     request.session['context'].append(response_text)

#     return HttpResponse(res)


@csrf_exempt
def auto_response(request):
    print('Session:', request.session['time'])
    template = loader.get_template('main/index.html')
    post = request.POST['post']

    uttr = post.split('_')[0]
    _personality = post.split('_')[1]
    rate = post.split('_')[2]

    file_name = 'records/' + request.session['time'] + '.txt'
    try:
        f = open(file_name, 'r')
        f.close()
    except:
        with open(file_name, 'w') as f:
            f.write(_personality + '\n')

    request.session['context'].append(uttr)

    print(request.session['context'])
    context_ = ' '.join(request.session['context'])

    user_emo = get_emo(uttr)
    user_personality = get_personality(uttr)
    user_intent = get_intent(uttr)

    personality = person_dict[_personality]
    response_emo = '' # emotionGenerationBERT(user_emo, uttr, personality)

    inputs = response_gen_tokenizer([context_], return_tensors="pt")
    reply_ids = response_gen_model.generate(**inputs)
    response_text = response_gen_tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]


    res_list = {
        'user_emo': emotion_mapping_2[user_emo],
        'response': response_text,  # '[' + response_emo + ']' +response_text,
        'user_personality': user_personality,
        'user_intent': user_intent,
        'response_emo': 'So'  # , I response you with in ' + response_emo + ':',
    }

    res = json.dumps(res_list)

    with open(file_name, 'a') as f:
        f.write(uttr + '\t' + response_text + '\t' + str(request.session['rate']) + '\n')

    request.session['rate'] = rate
    request.session['context'].append(response_text)

    return HttpResponse(res)


# ******************************************************************


def emotionGenerationBERT(user_emo, user_post, personality):
    if personality == 'A':
        response_emo_prob = np.array([i / sum(A_dict[user_emo]) for i in A_dict[user_emo]])
        index = np.random.choice([0, 1, 2, 3, 4, 5], p=response_emo_prob.ravel())
        response_emo = emo_dict[index]
    elif personality == 'C':
        response_emo_prob = np.array([i / sum(C_dict[user_emo]) for i in C_dict[user_emo]])
        index = np.random.choice([0, 1, 2, 3, 4, 5], p=response_emo_prob.ravel())
        response_emo = emo_dict[index]
    elif personality == 'E':
        response_emo_prob = np.array([i / sum(E_dict[user_emo]) for i in E_dict[user_emo]])
        index = np.random.choice([0, 1, 2, 3, 4, 5], p=response_emo_prob.ravel())
        response_emo = emo_dict[index]
    elif personality == 'O':
        response_emo_prob = np.array([i / sum(O_dict[user_emo]) for i in O_dict[user_emo]])
        index = np.random.choice([0, 1, 2, 3, 4, 5], p=response_emo_prob.ravel())
        response_emo = emo_dict[index]
    elif personality == 'N':
        response_emo_prob = np.array([i / sum(N_dict[user_emo]) for i in N_dict[user_emo]])
        index = np.random.choice([0, 1, 2, 3, 4, 5], p=response_emo_prob.ravel())
        response_emo = emo_dict[index]
    return response_emo


person_dict = {
    "Agreeable": "A",
    "Conscientious": "C",
    "Extroverted": "E",
    "Open": "O",
    "Neurotic": "N"
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


# =============


def test(test_dataloader, prompt_model):
    prompt_model.eval()
    labels_list = np.array([])
    pred_list = np.array([])
    overall_logits = []
    for step, inputs in enumerate(test_dataloader):
        inputs = inputs.cuda()
        logits = prompt_model(inputs)
        overall_logits.append(logits.detach())
    overall_logits = torch.cat(overall_logits)
    # print('='*20)
    # print(overall_logits.shape)
    return overall_logits


def get_personality(uttr):
    adam_epsilon = 1e-8
    num_class = 2

    method = 'Soft_Ours'
    datasets = 'Friends_Persona'
    SEED = 42
    use_cuda = True

    MAX_LEN = 128
    personalities = ['O', 'C', 'E', 'A', 'N']
    scores = []
    for personality in personalities:
        candidate_templates = []
        with open('personality_recognition_templates/Fine_Tuned_Friends_' + personality + '_SEED_' + str(
                SEED) + '_templates_top_10.txt', 'r') as f_template:
            candidate_templates = [i.strip() for i in f_template.readlines()]

        logits_all_templates = []

        for template in candidate_templates[:5]:
            mytemplate = ManualTemplate(
                text=template,
                tokenizer=tokenizer)
            wrapped_tokenizer = WrapperClass(max_seq_length=MAX_LEN, tokenizer=tokenizer, truncate_method="head")

            class_labels = [0, 1]

            with open('label_words/posterior_' + personality + '_label_words_SEED_' + str(SEED) + '.txt',
                      'r') as f_verbalizer:
                pos = [i.strip() for i in f_verbalizer.readline().split(',')]
                neg = [i.strip() for i in f_verbalizer.readline().split(',')]

            with open('label_words/posterior_' + personality + '_label_weights_SEED_' + str(SEED) + '.txt',
                      'r') as f_verbalizer:
                pos_weights = eval(f_verbalizer.readline())
                neg_weights = eval(f_verbalizer.readline())

            diff_len = len(neg_weights) - len(pos_weights)
            if diff_len >= 0:
                label_words_weights = torch.Tensor([neg_weights, pos_weights + [0] * diff_len])
            else:
                label_words_weights = torch.Tensor([neg_weights + [0] * (-diff_len), pos_weights])

            # print('label word weights: ', label_words_weights.shape)

            myverbalizer = ManualVerbalizer(
                classes=class_labels,
                label_words={
                    0: neg,
                    1: pos
                },
                tokenizer=tokenizer,
                label_words_weights=label_words_weights.cuda())

            prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer,
                                                   freeze_plm=False)
            prompt_model = prompt_model.cuda()

            sample = InputExample(text_a=uttr)

            test_dataloader = PromptDataLoader(
                dataset=[sample],
                template=mytemplate,
                tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass,
                max_seq_length=MAX_LEN,
                batch_size=1,
                shuffle=True,
                teacher_forcing=False,
                predict_eos_token=False,
                truncate_method="head")

            logits_ = test(test_dataloader, prompt_model)
        logits_all_templates.append(logits_.unsqueeze(dim=0))
        logits_all_templates = torch.cat(logits_all_templates)
        logits_all_templates = torch.mean(logits_all_templates, 0).squeeze(0)
        score = int(F.softmax(logits_all_templates)[1] * 100)
        scores.append(score)

    return scores








