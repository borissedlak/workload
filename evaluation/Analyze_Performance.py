import os.path
from os import path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = os.path.dirname(__file__)

boxplot_model_list = []
overall_list = pd.DataFrame({'Video#1 Model#1': []})


def analyze_performance(VIDEO, MODEL, M, FRAMES_COUNT):
    face_trigger_path = ROOT + f'/csv_export/function_time/{VIDEO}/{MODEL}/Face_Trigger.csv'
    age_trigger_path = ROOT + f'/csv_export/function_time/{VIDEO}/{MODEL}/Age_Trigger.csv'
    gender_trigger_path = ROOT + f'/csv_export/function_time/{VIDEO}/{MODEL}/Gender_Trigger.csv'
    blur_pixelate_path = ROOT + f'/csv_export/function_time/{VIDEO}/{MODEL}/Blur_Area_Pixelate.csv'
    fill_area_path = ROOT + f'/csv_export/function_time/{VIDEO}/{MODEL}/Fill_Area_Box.csv'
    overall_path = ROOT + f'/csv_export/function_time/{VIDEO}/{MODEL}/Overall_Chain.csv'

    fig, ax = plt.subplots()
    _dataframe = pd.DataFrame({'Face_Trigger': []})

    if path.exists(face_trigger_path):
        df_face_trigger = pd.read_csv(face_trigger_path)
        df_face_trigger['Face_Trigger'] = df_face_trigger['execution_time']
        df_face_trigger['timestamp'] = pd.to_datetime(df_face_trigger['timestamp'])
        df_face_trigger.plot(y="Face_Trigger", ax=ax, color='red', legend="abcd")
        plt.fill_between(range(0, FRAMES_COUNT), df_face_trigger['Face_Trigger'], step="pre", alpha=0.4,
                         color='red')
        _dataframe['Face_Trigger'] = df_face_trigger['Face_Trigger']

    if path.exists(age_trigger_path):
        df_age_trigger = pd.read_csv(age_trigger_path)
        df_age_trigger['Age_Trigger'] = df_age_trigger['execution_time']
        df_age_trigger['timestamp'] = pd.to_datetime(df_age_trigger['timestamp'])
        df_age_trigger.plot(y="Age_Trigger", ax=ax, color='blue', legend="abcd")
        plt.fill_between(range(0, FRAMES_COUNT), df_age_trigger['Age_Trigger'], step="pre", alpha=0.4, color='blue')
        _dataframe['Age_Trigger'] = df_age_trigger['Age_Trigger']

    if path.exists(gender_trigger_path):
        df_gender_trigger = pd.read_csv(gender_trigger_path)
        df_gender_trigger['Gender_Trigger'] = df_gender_trigger['execution_time']
        df_gender_trigger['timestamp'] = pd.to_datetime(df_gender_trigger['timestamp'])
        df_gender_trigger.plot(y="Gender_Trigger", ax=ax, color='blue', legend="abcd")
        plt.fill_between(range(0, FRAMES_COUNT), df_gender_trigger['Gender_Trigger'], step="pre", alpha=0.4,
                         color='blue')
        _dataframe['Gender_Trigger'] = df_gender_trigger['Gender_Trigger']

    if path.exists(blur_pixelate_path):
        df_blur_pixelate = pd.read_csv(blur_pixelate_path)
        df_blur_pixelate['Blur_Pixelate'] = df_blur_pixelate['execution_time']
        df_blur_pixelate['timestamp'] = pd.to_datetime(df_blur_pixelate['timestamp'])
        df_blur_pixelate.plot(y="Blur_Pixelate", ax=ax, color='green')
        plt.fill_between(range(0, FRAMES_COUNT), df_blur_pixelate['Blur_Pixelate'], step="pre", alpha=0.35,
                         color='green')
        _dataframe['Blur_Pixelate'] = df_blur_pixelate['Blur_Pixelate']

    if path.exists(fill_area_path):
        df_fill_area = pd.read_csv(fill_area_path)
        df_fill_area['Fill_Area'] = df_fill_area['execution_time']
        df_fill_area['timestamp'] = pd.to_datetime(df_fill_area['timestamp'])
        df_fill_area.plot(y="Fill_Area", ax=ax, color='green')
        plt.fill_between(range(0, FRAMES_COUNT), df_fill_area['Fill_Area'], step="pre", alpha=0.35, color='green')
        _dataframe['Fill_Area'] = df_fill_area['Fill_Area']

    if path.exists(overall_path):
        df_overall = pd.read_csv(overall_path)
        df_overall['Overall'] = df_overall['execution_time']
        df_overall['timestamp'] = pd.to_datetime(df_overall['timestamp'])
        df_overall.plot(y="Overall", ax=ax, color='black')
        plt.fill_between(range(0, FRAMES_COUNT), df_overall['Overall'], step="pre", alpha=0.1, color='black')
        _dataframe['Overall'] = df_overall['Overall']
        overall_list[f"{VIDEO.replace('_', '#').replace('v', 'V')} {MODEL.replace('_', '#').replace('m', 'M')}"] = \
            df_overall['Overall']

    ax.legend(loc=1)
    ax.set_xlabel("Index of Frame in Stream")
    ax.set_ylabel("Milliseconds (ms)")
    ax.set_title(f"Performance of SFU processing {VIDEO.replace('_', '#')} with {MODEL.replace('_', '#')}")
    fig.savefig(ROOT + f'/figures/{VIDEO}/{MODEL}/performance_{VIDEO}_{MODEL}.png', bbox_inches='tight')
    fig.show()

    if M == 1:
        _dataframe = _dataframe[['Face_Trigger', 'Blur_Pixelate', 'Overall']]
    elif M == 2:
        _dataframe = _dataframe[['Face_Trigger', 'Age_Trigger', 'Blur_Pixelate', 'Overall']]
    elif M == 3:
        _dataframe = _dataframe[['Face_Trigger', 'Gender_Trigger', 'Blur_Pixelate', 'Overall']]
    elif M == 4:
        _dataframe = _dataframe[['Face_Trigger', 'Fill_Area', 'Overall']]

    model_x = _dataframe.assign(Model=M)
    boxplot_model_list.append(model_x)


##########################################################################

analyze_performance('video_1', 'model_1', 1, 302)
analyze_performance('video_1', 'model_2', 2, 302)
analyze_performance('video_1', 'model_3', 3, 302)
analyze_performance('video_1', 'model_4', 4, 302)

cdf = pd.concat(boxplot_model_list)
mdf = pd.melt(cdf, id_vars=['Model'], var_name=['Letter'])

ax = sns.boxplot(x="Model", y="value", hue="Letter", data=mdf)
ax.legend(loc=1)
ax.set_title("Individual performance for Video #1")
ax.set_xlabel("Index of Model")
ax.set_ylabel("Milliseconds (ms)")
plt.savefig(ROOT + f'/figures/video_1/performance_box_video_1.png', bbox_inches='tight')
plt.show()

ax = overall_list.plot()
ax.legend(loc=1)
ax.set_title("Individual performance for Video #1")
ax.set_xlabel("Index of Frame in Stream")
ax.set_ylabel("Milliseconds (ms)")
plt.savefig(ROOT + f'/figures/video_1/performance_overall_video_1.png', bbox_inches='tight')
plt.show()

overall_list = pd.DataFrame({'Video#2 Model#1': []})
boxplot_model_list = []

analyze_performance('video_2', 'model_1', 1, 163)
analyze_performance('video_2', 'model_2', 2, 163)
analyze_performance('video_2', 'model_3', 3, 163)
analyze_performance('video_2', 'model_4', 4, 163)

cdf = pd.concat(boxplot_model_list)
mdf = pd.melt(cdf, id_vars=['Model'], var_name=['Letter'])

ax = sns.boxplot(x="Model", y="value", hue="Letter", data=mdf)
ax.legend(loc=1)
ax.set_title("Individual performance for Video #2")
ax.set_xlabel("Index of Model")
ax.set_ylabel("Milliseconds (ms)")
plt.savefig(ROOT + f'/figures/video_2/performance_box_video_2.png', bbox_inches='tight')
plt.show()

ax = overall_list.plot()
ax.legend(loc=1)
ax.set_title("Overall performances for Video #2")
ax.set_xlabel("Index of Frame in Stream")
ax.set_ylabel("Milliseconds (ms)")
plt.savefig(ROOT + f'/figures/video_2/performance_overall_video_2.png', bbox_inches='tight')
plt.show()
