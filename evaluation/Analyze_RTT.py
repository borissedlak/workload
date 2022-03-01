import os

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(__file__)

df_laptop = pd.read_csv(ROOT + '/csv_export/rtt/consumer_rtt_laptop_consumer.csv')
df_phone = pd.read_csv(ROOT + '/csv_export/rtt/consumer_rtt_phone_consumer.csv')
df_same = pd.read_csv(ROOT + '/csv_export/rtt/consumer_rtt_same_device.csv')
df_laptop['timestamp'] = pd.to_datetime(df_laptop['timestamp'])
df_phone['timestamp'] = pd.to_datetime(df_phone['timestamp'])
df_same['timestamp'] = pd.to_datetime(df_same['timestamp'])

df_laptop['rtt'] = df_laptop['rtt'] * 1000
df_phone['rtt'] = df_phone['rtt'] * 1000
df_same['rtt'] = df_same['rtt'] * 1000

phone_mean = df_phone['rtt'].mean()
laptop_mean = df_laptop['rtt'].mean()
same_mean = df_same['rtt'].mean()

fig, ax = plt.subplots()

df_phone.plot(y="rtt", ax=ax, color='#2b5a1d')
df_laptop.plot(y="rtt", ax=ax, color='#7e0060')
df_same.plot(y="rtt", ax=ax, color='orange')

ax.legend(["Phone", "Laptop", "Local"])
ax.set_xlabel("Total time in s")
ax.set_ylabel("Milliseconds (ms)")
ax.set_title("Latency from SFU \u2192 Consumer")
fig.savefig(ROOT + '/figures/latency_consumer.png', bbox_inches='tight')
fig.show()

df_bar = pd.DataFrame({'lab': ['Phone', 'Laptop', 'Local'], 'val': [phone_mean, laptop_mean, same_mean]})
ax = df_bar.plot.bar(x='lab', y='val', rot=0, color=['#2b5a1d', '#7e0060', 'orange'], legend=False)
ax.set_xlabel(" ")
ax.set_title("Mean Latency from SFU \u2192 Consumer")
ax.get_figure().savefig(ROOT + '/figures/mean_latency_consumer.png', bbox_inches='tight')
ax.get_figure().show()

##########################################################################


df_aws = pd.read_csv(ROOT + '/csv_export/rtt/producer_rtt_aws_europe.csv')
df_edge = pd.read_csv(ROOT + '/csv_export/rtt/producer_rtt_edge_local.csv')
df_aws['timestamp'] = pd.to_datetime(df_aws['timestamp'])
df_edge['timestamp'] = pd.to_datetime(df_edge['timestamp'])

df_aws['rtt'] = df_aws['rtt'] * 1000
df_edge['rtt'] = df_edge['rtt'] * 1000

aws_mean = df_aws['rtt'].mean()
edge_mean = df_edge['rtt'].mean()

fig, ax = plt.subplots()

df_aws.plot(y="rtt", ax=ax, color='#47abd8')
df_edge.plot(y="rtt", ax=ax, color='#D01B1B')

ax.legend(["Cloud", "Edge"])
ax.set_xlabel("Total time in s")
ax.set_ylabel("Milliseconds (ms)")
ax.set_title("Latency from Producer \u2192 SFU")
fig.savefig(ROOT + '/figures/latency_producer.png', bbox_inches='tight')
fig.show()

df_bar = pd.DataFrame({'lab': ['Cloud', 'Edge'], 'val': [aws_mean, edge_mean]})
ax = df_bar.plot.bar(x='lab', y='val', rot=0, color=['#47abd8', '#D01B1B'], legend=False)
ax.set_xlabel(" ")
ax.set_title("Mean Latency from Producer \u2192 SFU")
ax.get_figure().savefig(ROOT + '/figures/mean_latency_producer.png', bbox_inches='tight')
ax.get_figure().show()
