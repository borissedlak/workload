import os.path
from os import path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(__file__)


def analyze_performance(VIDEO, MODEL, FRAMES_COUNT):
    face_trigger_path = ROOT + f'/csv_export/function_time/{VIDEO}/{MODEL}/Face_Trigger.csv'
    age_trigger_path = ROOT + f'/csv_export/function_time/{VIDEO}/{MODEL}/Age_Trigger.csv'
    gender_trigger_path = ROOT + f'/csv_export/function_time/{VIDEO}/{MODEL}/Gender_Trigger.csv'
    blur_pixelate_path = ROOT + f'/csv_export/function_time/{VIDEO}/{MODEL}/Blur_Area_Pixelate.csv'
    fill_area_path = ROOT + f'/csv_export/function_time/{VIDEO}/{MODEL}/Fill_Area_Box.csv'
    overall_path = ROOT + f'/csv_export/function_time/{VIDEO}/{MODEL}/Overall_Chain.csv'

    fig, ax = plt.subplots()

    if path.exists(face_trigger_path):
        df_face_trigger = pd.read_csv(face_trigger_path)
        df_face_trigger['Face_Trigger'] = df_face_trigger['execution_time']
        df_face_trigger['timestamp'] = pd.to_datetime(df_face_trigger['timestamp'])
        df_face_trigger.plot(y="Face_Trigger", ax=ax, color='red', legend="abcd")
        plt.fill_between(range(0, FRAMES_COUNT), df_face_trigger['Face_Trigger'], step="pre", alpha=0.3, color='red')

    if path.exists(age_trigger_path):
        df_age_trigger = pd.read_csv(age_trigger_path)
        df_age_trigger['Age_Trigger'] = df_age_trigger['execution_time']
        df_age_trigger['timestamp'] = pd.to_datetime(df_age_trigger['timestamp'])
        df_age_trigger.plot(y="Age_Trigger", ax=ax, color='blue', legend="abcd")
        plt.fill_between(range(0, FRAMES_COUNT), df_age_trigger['Age_Trigger'], step="pre", alpha=0.3, color='blue')

    if path.exists(gender_trigger_path):
        df_gender_trigger = pd.read_csv(gender_trigger_path)
        df_gender_trigger['Gender_Trigger'] = df_gender_trigger['execution_time']
        df_gender_trigger['timestamp'] = pd.to_datetime(df_gender_trigger['timestamp'])
        df_gender_trigger.plot(y="Gender_Trigger", ax=ax, color='blue', legend="abcd")
        plt.fill_between(range(0, FRAMES_COUNT), df_gender_trigger['Gender_Trigger'], step="pre", alpha=0.3, color='blue')

    if path.exists(blur_pixelate_path):
        df_blur_pixelate = pd.read_csv(blur_pixelate_path)
        df_blur_pixelate['Blur_Pixelate'] = df_blur_pixelate['execution_time']
        df_blur_pixelate['timestamp'] = pd.to_datetime(df_blur_pixelate['timestamp'])
        df_blur_pixelate.plot(y="Blur_Pixelate", ax=ax, color='green')
        plt.fill_between(range(0, FRAMES_COUNT), df_blur_pixelate['Blur_Pixelate'], step="pre", alpha=0.35, color='green')

    if path.exists(fill_area_path):
        df_fill_area = pd.read_csv(fill_area_path)
        df_fill_area['Fill_Area'] = df_fill_area['execution_time']
        df_fill_area['timestamp'] = pd.to_datetime(df_fill_area['timestamp'])
        df_fill_area.plot(y="Fill_Area", ax=ax, color='green')
        plt.fill_between(range(0, FRAMES_COUNT), df_fill_area['Fill_Area'], step="pre", alpha=0.35, color='green')

    if path.exists(overall_path):
        df_overall = pd.read_csv(overall_path)
        df_overall['Overall'] = df_overall['execution_time']
        df_overall['timestamp'] = pd.to_datetime(df_overall['timestamp'])
        df_overall.plot(y="Overall", ax=ax, color='black')
        plt.fill_between(range(0, FRAMES_COUNT), df_overall['Overall'], step="pre", alpha=0.1, color='black')

    ax.legend(loc=1)
    ax.set_xlabel("Total time in s")
    ax.set_ylabel("Milliseconds (ms)")
    ax.set_title(f"Performance of SFU processing {VIDEO.replace('_', '#')} with {MODEL.replace('_', '#')}")
    fig.savefig(ROOT + f'/figures/{VIDEO}/{MODEL}/performance_{VIDEO}_{MODEL}.png', bbox_inches='tight')
    fig.show()


##########################################################################

analyze_performance('video_1', 'model_1', 302)
analyze_performance('video_1', 'model_2', 302)
analyze_performance('video_1', 'model_3', 302)
analyze_performance('video_1', 'model_4', 302)
analyze_performance('video_2', 'model_1', 163)
analyze_performance('video_2', 'model_2', 163)
analyze_performance('video_2', 'model_3', 163)
analyze_performance('video_2', 'model_4', 163)
