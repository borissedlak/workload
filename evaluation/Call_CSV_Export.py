import time

import requests

sfu_url = "http://192.168.0.80:4000"
sfu_get_stats = sfu_url + "/calculate_stats"
sfu_persist_stats = sfu_url + "/persist_stats"

producer_url = "http://192.168.0.101:8081/provide"
producer_get_stats = sfu_url + "/calculate_stats"
producer_persist_stats = sfu_url + "/persist_stats"


for i in range(60):
    print(f"{i+1}. Round")
    consumer_ret = requests.get(sfu_get_stats, timeout=1.0).content
    producer_ret = requests.get(producer_get_stats, timeout=1.0).content
    print(consumer_ret)
    print(producer_ret)
    print("\n")
    time.sleep(1)



consumer_ret = requests.post(sfu_persist_stats, timeout=1.0).content
producer_ret = requests.post(producer_persist_stats, timeout=1.0).content
print(consumer_ret)
print(producer_ret)
