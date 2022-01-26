import time

import requests

sfu_url = "http://192.168.0.80:4000"
sfu_get_stats = sfu_url + "/calculate_stats"
sfu_persist_stats = sfu_url + "/persist_stats"

producer_url = "http://192.168.0.101:8081"
producer_get_stats = producer_url + "/calculate_stats"
producer_persist_stats = producer_url + "/persist_stats"

for i in range(60):
    print(f"{i + 1}. Round")
    consumer_ret = requests.get(sfu_get_stats, timeout=1.0).content
    producer_ret = requests.get(producer_get_stats, timeout=1.0).content
    print(f"Consumer: {consumer_ret}")
    print(f"Producer: {producer_ret}")
    print("\n")
    time.sleep(1)


print("Persisting RTTs to CSV")
consumer_ret = requests.post(sfu_persist_stats, timeout=1.0).content
producer_ret = requests.post(producer_persist_stats, timeout=1.0).content
print(f"Consumer: {consumer_ret}")
print(f"Producer: {producer_ret}")
