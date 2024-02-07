from transformers import AutoModel, AutoTokenizer
# import py_vncorenlp
import torch
from model_arch import BERT_Arch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import random

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")

device = torch.device("cpu")

model = BERT_Arch(phobert)
model_path = 'models/my_model.pth'  
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
# print(checkpoint.keys())
model.load_state_dict(checkpoint)

le = LabelEncoder()
df = pd.read_excel('intents.xlsx')
df['label'] = le.fit_transform(df['label'])
data = pd.read_json('intents.json')

max_seq_len = 10

def get_prediction(str):
#  str = rdrsegmenter.word_segment(str)
 test_text = [str]
 model.eval()

 tokens_test_data = tokenizer(
 test_text,
 max_length = max_seq_len,
 pad_to_max_length=True,
 truncation=True,
 return_token_type_ids=False
 )
 test_seq = torch.tensor(tokens_test_data['input_ids'])
 test_mask = torch.tensor(tokens_test_data['attention_mask'])

 preds = None
 with torch.no_grad():
   preds = model(test_seq.to(device), test_mask.to(device))
 preds = preds.detach().cpu().numpy()
 preds = np.argmax(preds, axis = 1)
 print("Intent Identified: ", le.inverse_transform(preds)[0])
 return le.inverse_transform(preds)[0]

def get_response(message):
  intent = get_prediction(message)
  if (intent=="Weather"):
    min_temp = get_weather()['DailyForecasts'][0]['Temperature']['Minimum']['Value']
    max_temp = get_weather()['DailyForecasts'][0]['Temperature']['Maximum']['Value']
    today_condition = get_weather()['DailyForecasts'][0]['Day']['IconPhrase']
    important_prediction = get_weather()['Headline']['Text']
    final_result = f"Dự báo thời tiết: Nhiệt độ từ {min_temp} đến {max_temp} độ C, ban ngày {today_condition}, {important_prediction}"
    return final_result
  for i in data['intents']:
    if i["label"] == intent:
      result = random.choice(i["answers"])
      final_result = f"{result}"
      return final_result
    

import requests
def get_weather():
  api_key = 'AXb2NpRDDbx12ws6IZGGCeEGDJFwpRw7'

  # Make a GET request to the OpenWeatherMap API
  url = f'http://dataservice.accuweather.com/forecasts/v1/daily/1day/353412?apikey=AXb2NpRDDbx12ws6IZGGCeEGDJFwpRw7&language=vi-vn&metric=true'
  response = requests.get(url)

  # Parse the JSON response
  data = response.json()
  # description = data['weather'][0]['description']
  return data

# get_response("Thời tiết hôm nay thế nào hả mày?")