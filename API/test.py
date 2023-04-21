import requests
import json
with open('C:/Users/User/Desktop/Хакатон/hackathon0423/ALGORITM/data_for_algoritm_14946.json') as json_file:
    params = json.load(json_file)

res = requests.post("http://127.0.0.1:3000/api/model", json = params)

print(res.text)