import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import os


def merge_groups(path = '/media/annette/TOSHIBA_EXT/DLC Project/results/validation'):

    mouse_name = []
    condition = []
    peak_frame = []
    depth = []
    start_frame = []
    end_frame = []
    bodypart = []

    mouse_name_summary = []
    condition_summary = []
    n_slips_summary = []
    mean_depth_summary = []
    sd_depth_summary = []
    mean_duration_summary = []
    sd_duration_summary = []

    files = os.listdir(path)

    for file in files:
        file_name = os.path.join(path, file)
        mouse = file.split('_')[0]
        cond = file.split('_')[1]
        df = pd.read_csv(file_name)
        
        # mouse id and experiment type
        for i in range(len(df)):
            mouse_name.append(mouse)
            condition.append(cond)
        # detailed results
        peak_frame.extend(df['time'])
        depth.extend(df['depth'])
        start_frame.extend(df['start'])
        end_frame.extend(df['end'])
        bodypart.extend(df['bodypart'])
        
        # summary
        mouse_name_summary.append(mouse)
        condition_summary.append(cond)
        
        n_slips = len(df)
        if n_slips == 0:
            mean_depth = np.nan
            sd_depth = np.nan
            mean_duration = np.nan
            sd_duration = np.nan
        else:
            mean_depth = np.mean(df['depth'])
            sd_depth = np.std(df['depth'])
            mean_duration = np.mean(df['end'] - df['start'])
            sd_duration = np.std(df['end'] - df['start'])
        
        n_slips_summary.append(n_slips)
        mean_depth_summary.append(mean_depth)
        sd_depth_summary.append(sd_depth)
        mean_duration_summary.append(mean_duration)
        sd_duration_summary.append(sd_duration)

        results = pd.DataFrame({'mouse_name':mouse_name,
                        'condition':condition,
                        'peak_frame':peak_frame,
                        'depth': depth,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'bodypart': bodypart
                       })

        summary = pd.DataFrame({'mouse_name':mouse_name_summary,
                        'condition':condition_summary,
                        'n_slips':n_slips_summary,
                        'mean_depth':mean_depth_summary,
                        'sd_depth': sd_depth_summary,
                        'mean_duration': mean_duration_summary,
                        'sd_duration': sd_duration_summary
                       })

        return results, summary


def t_test(control_file, exp_file, variable = 'n_slips'):
    t, p = ttest_ind(control_file, exp_file, variable)
    return t, p