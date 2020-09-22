# Input: data/wtwt_ids.json and data/scrapped_full/*
# Output: data/pre_processed.json

# Read all existing scrapped files and preprocess, tokenize the full_text part
#   Read wtwt_ids.json and for each tweet_id, append toked_text. Save into pre_processed.json

import json
import os
import sys
import glob

import re
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons


FILES = ['can_not_test-add_text.jsonl', 'cure_and_prevention-add_text.jsonl',
         'death-add_text.jsonl', 'positive-add_text.jsonl', 'negative-add_text.jsonl']
FILES = ['../data/' + FILE for FILE in FILES

EMOJI_PATTERN = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},

    fix_html=True,
    segmenter="twitter",
    corrector="twitter",

    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=True,

    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons]
)

REMOVE_TAGS = [
    "<emphasis>", "<kiss>", "<repeated>", "<laugh>", "<allcaps>",
    "</allcaps>", "<angel>", "<elongated>", "<tong>", "<annoyed>",
    "<censored>", "<happy>", "<percent>", "<wink>",
    "<headdesk>", "<surprise>", "<date>", "<time>", "<url>",
    "<sad>", "<email>", "<phone>", "<hashtag>", "</hashtag>",
    "<number>", "<money>"
    ]

ADD_TO_GLOVE = ["<user>"]

PUNCTS = '''()-[]{;}\,<>/@#'%"^*_~.?!| +:=`'''

def decontracted(phrase):
  phrase = re.sub(r"won\'t", "will not", phrase)
  phrase = re.sub(r"can\'t", "can not", phrase)

  phrase = re.sub(r"n\'t", " not", phrase)
  phrase = re.sub(r"\'re", " are", phrase)
  phrase = re.sub(r"\'s", " is", phrase)
  phrase = re.sub(r"\'d", " would", phrase)
  phrase = re.sub(r"\'ll", " will", phrase)
  phrase = re.sub(r"\'t", " not", phrase)
  phrase = re.sub(r"\'ve", " have", phrase)
  phrase = re.sub(r"\'m", " am", phrase)
  phrase = re.sub(r"\'m", " am", phrase)
  phrase = re.sub(r"1st", " first ", phrase)
  phrase = re.sub(r"2nd", " second ", phrase)
  phrase = re.sub(r"3rd", " third ", phrase)
  phrase = re.sub(r"—", " ", phrase)
  phrase = re.sub(r"-", " ", phrase)

  return phrase

def pre_process_single(tweet):
  tweet_toked_text = []
  de_emojified_text = tweet.encode('ascii', 'ignore').decode('ascii')
  de_emojified_text = EMOJI_PATTERN.sub(r' ', de_emojified_text)
  de_emojified_text = decontracted(de_emojified_text)
  company_normalize_text = de_emojified_text

  tokens = text_processor.pre_process_doc(company_normalize_text)
  for token in tokens:
    if token in REMOVE_TAGS:
      pass
    else:
      tweet_toked_text.append(token)     
  if len(tweet_toked_text) < 1:
    pass#print(tweet, tokens, t_id)
  return tweet_toked_text

for file in FILES:
  savepath = "preproced_" + file[:-1]
  fin = open(file, 'r')
  lines = fin.read().split("\n")
  event_data = []
  for line in lines:
    if line.strip() == "":
      continue
    j = json.loads(line)
    j['prepro'] = " ".join(pre_process_single(j['text']))
    event_data.append(j.copy())
  fin.close()
  fout = open(savepath, "w+")
  json.dump(event_data, fout)
  fout.close()

