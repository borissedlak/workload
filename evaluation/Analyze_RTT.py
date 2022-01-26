import os

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(__file__)

df_laptop = pd.read_csv(ROOT + '/csv_export/consumer_rtt_laptop_consumer.csv')
df_phone = pd.read_csv(ROOT + '/csv_export/consumer_rtt_phone_consumer.csv')
df_same = pd.read_csv(ROOT + '/csv_export/consumer_rtt_same_device.csv')
df_laptop['timestamp'] = pd.to_datetime(df_laptop['timestamp'])
df_phone['timestamp'] = pd.to_datetime(df_phone['timestamp'])
df_same['timestamp'] = pd.to_datetime(df_same['timestamp'])

print(df_phone['rtt'].mean())
print(df_laptop['rtt'].mean())
print(df_same['rtt'].mean())

fig, ax = plt.subplots()

df_phone.plot(y="rtt", ax=ax)
df_laptop.plot(y="rtt", ax=ax)
df_same.plot(y="rtt", ax=ax)

ax.legend(["Phone", "Laptop", "Local"])
fig.savefig(ROOT + '/figures/rtt_edge.png')
