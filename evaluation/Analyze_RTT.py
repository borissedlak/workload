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

phone_mean = df_phone['rtt'].mean()
laptop_mean = df_laptop['rtt'].mean()
same_mean = df_same['rtt'].mean()

fig, ax = plt.subplots()

df_phone.plot(y="rtt", ax=ax, color='blue')
df_laptop.plot(y="rtt", ax=ax, color='red')
df_same.plot(y="rtt", ax=ax, color='orange')

ax.legend(["Phone", "Laptop", "Local"])
ax.set_xlabel("Total time in s")
ax.set_ylabel("RTT SFU \u2192 Consumer in ms")
ax.set_title("RTT from SFU \u2192 Consumer")
fig.savefig(ROOT + '/figures/rtt_edge.png')
fig.show()
