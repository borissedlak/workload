import os.path
from os import path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(__file__)
VIDEO = 'video#1'
MODEL = 'model#1'

face_trigger_path = ROOT + f'/csv_export/function_time/{VIDEO}/{MODEL}/Face_Trigger.csv'
blur_pixelate_path = ROOT + f'/csv_export/function_time/{VIDEO}/{MODEL}/Blur_Area_Pixelate.csv'
overall_path = ROOT + f'/csv_export/function_time/{VIDEO}/{MODEL}/Overall_Chain.csv'

fig, ax = plt.subplots()
legend_list = []

if path.exists(face_trigger_path):
    df_face_trigger = pd.read_csv(face_trigger_path)
    df_face_trigger['Face_Trigger'] = df_face_trigger['execution_time']
    df_face_trigger['timestamp'] = pd.to_datetime(df_face_trigger['timestamp'])
    df_face_trigger.plot(y="Face_Trigger", ax=ax, color='#2b5a1d', legend="abcd")
    plt.fill_between(range(0, 169), df_face_trigger['Face_Trigger'], step="pre", alpha=0.2, color='#2b5a1d')
if path.exists(blur_pixelate_path):
    df_blur_pixelate = pd.read_csv(blur_pixelate_path)
    df_blur_pixelate['Blur_Pixelate'] = df_blur_pixelate['execution_time']
    df_blur_pixelate['timestamp'] = pd.to_datetime(df_blur_pixelate['timestamp'])
    df_blur_pixelate.plot(y="Blur_Pixelate", ax=ax, color='#7e0060')
    plt.fill_between(range(0, 169), df_blur_pixelate['Blur_Pixelate'], step="pre", alpha=0.3, color='#7e0060')
if path.exists(overall_path):
    df_overall = pd.read_csv(overall_path)
    df_overall['Overall'] = df_overall['execution_time']
    df_overall['timestamp'] = pd.to_datetime(df_overall['timestamp'])
    df_overall.plot(y="Overall", ax=ax, color='black')
    plt.fill_between(range(0, 169), df_overall['Overall'], step="pre", alpha=0.1, color='black')

ax.legend(loc=1)
ax.set_xlabel("Total time in s")
ax.set_ylabel("Milliseconds (ms)")
ax.set_title(f"Performance of SFU processing {VIDEO} with {MODEL}")
fig.savefig(ROOT + f'/figures/{VIDEO}/{MODEL}/function_performance.png', bbox_inches='tight')
fig.show()

##########################################################################
