    '''
    pd_dataframe: contains raw coordinates (not adjusted for treadmill movement, unfiltered)
    bodypart: which bodypart to use for stride estimation
    '''
    starts_all = []
    ends_all = []
    durations = []
    limbs = []
        
    cycle_dur_secs = []
    cycle_dur_frames = []
    cycle_vs = []
    
    stride_lens = []
    stance_dur_secs = []
    swing_dur_secs = []
    swing_percentages = []
    stance_percentages = []

    limb_len_means = []
    limb_len_maxs = []
    limb_len_mins = []
    limb_len_sds = []
    
    step_heights = []
    max_v_during_swings = []
    
    mtp_joint_extensions = []
    ankle_joint_extensions = []
    knee_joint_extensions = []
    hip_joint_extensions = []
    
    mtp_joint_flexions = []
    ankle_joint_flexions = []
    knee_joint_flexions = []
    hip_joint_flexions = []
    
    mtp_joint_amplitudes = []
    ankle_joint_amplitudes = []
    knee_joint_amplitudes = []
    hip_joint_amplitudes = []
    
    mtp_joint_sds = []
    ankle_joint_sds = []
    knee_joint_sds = []
    hip_joint_sds = []
    
    drag_ts = []
    drag_percentages = []

    dtw_x_plane_5_means = [] # dynamic time warping: measures spatial variability
    dtw_x_plane_5_sds = []
    dtw_y_plane_5_means = []
    dtw_y_plane_5_sds = []
    dtw_xy_plane_5_means = []
    dtw_xy_plane_5_sds = []

    dtw_x_plane_10_means = []
    dtw_x_plane_10_sds = []
    dtw_y_plane_10_means = []
    dtw_y_plane_10_sds = []
    dtw_xy_plane_10_means = []
    dtw_xy_plane_10_sds = []
    

#     px_speed, _ = scipy.stats.norm.fit(x_change_filt[x_change_filt>0])

#     if cm_speed is None: 
#         cm_speed = px_speed / px_to_cm_speed_ratio

#     pixels_per_cm = 45/2.54

#     bodyparts = ['toe', 'mtp', 'ankle', 'knee', 'hip', 'iliac crest']
#     pd_dataframe = treadmill_correction(pd_dataframe, bodyparts, px_speed)
    for num, direction in enumerate(['L', 'R']):
        
        x_change = np.diff(pd_dataframe[f'toe{direction} x'][pd_dataframe[f'toe{direction} likelihood']>0.5])
        x_change_filt = x_change[(x_change < np.mean(x_change) + 1*np.std(x_change)) & (x_change > np.mean(x_change) - 1*np.std(x_change))]

        smooth_toe_x = butterworth_filter(pd_dataframe[f'toe{direction} x'], frame_rate, cutoff_f)
        smooth_mtp_x = butterworth_filter(pd_dataframe[f'mtp{direction} x'], frame_rate, cutoff_f)
        smooth_ankle_x = butterworth_filter(pd_dataframe[f'ankle{direction} x'], frame_rate, cutoff_f)
        smooth_knee_x = butterworth_filter(pd_dataframe[f'knee{direction} x'], frame_rate, cutoff_f)
        smooth_hip_x = butterworth_filter(pd_dataframe[f'hip{direction} x'], frame_rate, cutoff_f)
        smooth_crest_x = butterworth_filter(pd_dataframe[f'crest{direction} x'], frame_rate, cutoff_f)

        smooth_toe_y = butterworth_filter(pd_dataframe[f'toe{direction} y'], frame_rate, cutoff_f)
        smooth_mtp_y = butterworth_filter(pd_dataframe[f'mtp{direction} y'], frame_rate, cutoff_f)
        smooth_ankle_y = butterworth_filter(pd_dataframe[f'ankle{direction} y'], frame_rate, cutoff_f)
        smooth_knee_y = butterworth_filter(pd_dataframe[f'knee{direction} y'], frame_rate, cutoff_f)
        smooth_hip_y = butterworth_filter(pd_dataframe[f'hip{direction} y'], frame_rate, cutoff_f)
        smooth_crest_y = butterworth_filter(pd_dataframe[f'crest{direction} y'], frame_rate, cutoff_f)


            # bodypart x coordinate rate of change does not meet assumption of multimodal distribution
            # instead it is centered around 0
            # step cycle cannot be detected properly

        else:
            starts = []
            ends = []

#             filtered_bodypart_x = butterworth_filter(pd_dataframe[f'toe{direction} x'], frame_rate, 5)
#             filtered_bodypart_x_change = np.diff(filtered_bodypart_x)

            bodypart_x_change = np.diff(df[f'toe{direction} x'])

            # left hind limb: walking right to left, x-coord decreasing during swing phase
            # is_swing = [i<0 for i in bodypart_x_change]
            if direction == 'L':
                is_stance = [i <= 0 for i in bodypart_x_change]
            else:
                is_stance = [i >= 0 for i in bodypart_x_change]
            for i in range(1, len(is_stance)):
                if is_stance[i] != is_stance[i-1]:
                    if is_stance[i] and df[f'toe{direction} likelihood'][i] > 0.6:
                        if len(starts) == 0:
                            starts.append(i)
                        elif abs(df[f'toe{direction} x'][starts[-1]] - df[f'toe{direction} x'][i])>20:
                            starts.append(i)
            for i in range(1,len(starts)):
                ends.append(starts[i]-1)
                
            ends.append(len(is_stance))
            
            starts_all.extend(starts)
            ends_all.extend(ends)

            y = pd_dataframe[f'toe{direction} y'][pd_dataframe[f'toe{direction} likelihood'] > 0.5]
            y_filt = y[(y < np.mean(y) + 1*np.std(y)) & (y > np.mean(y) - 1*np.std(y))]
            _, b = np.histogram(y_filt, bins=100, density=True)
            # fit 2 Gaussians to the pdf of toe y coord
            gm = GaussianMixture(n_components=2, random_state=0).fit(np.array(b).reshape(-1,1))
            # the Gaussian with larger mean corresponds to y coord during stance phase
            # use this mean as threshold to detect dragging during swing phase
            stance_threshold = max(gm.means_)
            # use this plus SD (i.e., lower in space) as treadmill y coord, to calculate step height
            treadmill_y = float(stance_threshold + np.sqrt(gm.covariances_[np.where(gm.means_ == stance_threshold)]))

            starts_included = []
            ends_included = []

            for i in range(len(starts)):
                limbs.append(direction)

                stride_len = abs(find_stride_len(smooth_toe_x, starts[i], ends[i])) # in pixel
                stride_len_cm = stride_len / pixels_per_cm
                limb_len_max = max(limb_lens[starts[i] : ends[i]]) / pixels_per_cm
#                 limb_len_max = max(limb_lens[starts[i] : ends[i]])
                cycle_dur_frame = ends[i] - starts[i]
                cycle_v = stride_len_cm / cycle_dur_frame * frame_rate # (px / frames) * (frames/s) = px/s

                stance_dur_frame, swing_perc, stance_perc = find_swing_stance(elevation_angle_change, starts[i], ends[i])
                swing_dur_sec = (cycle_dur_frame - stance_dur_frame) / frame_rate

                if swing_dur_sec and stance_dur_frame > 0:
                    # min: image y axis starts from top
#                     step_height = -(min(smooth_toe_y[starts[i]+stance_dur_frame:ends[i]]) - \
#                                     max(smooth_toe_y[starts[i]:starts[i]+stance_dur_frame])) / pixels_per_cm
                    step_height = -(min(smooth_toe_y[starts[i]:ends[i]]) - \
                                    max(smooth_toe_y[starts[i]:ends[i]])) / pixels_per_cm
                elif swing_dur_sec and stance_dur_frame == 0:
                    step_height = -(min(smooth_toe_y[starts[i]+stance_dur_frame:ends[i]]) - \
                                    smooth_toe_y[starts[i]]) / pixels_per_cm
                else:
                    # no swing phase!
                    step_height = 0


                # does extracted stride seem accurate? (e.g., due to mislabeled bodypart position)
                # extreme limb length
                # extreme stride length
                # single frame "cycle" duration
                # extreme step height
                
                if no_outlier_filter or (limb_len_max < 15 and stride_len_cm < 8 and \
                    cycle_dur_frame > 1 and step_height < 1.5 and step_height > 0):

                    starts_included.append(starts[i])
                    ends_included.append(ends[i])

                    limb_len_means.append(np.mean(limb_lens[starts[i] : ends[i]]) / pixels_per_cm)
                    limb_len_maxs.append(limb_len_max)
                    limb_len_mins.append(min(limb_lens[starts[i] : ends[i]]) / pixels_per_cm)
                    limb_len_sds.append(np.std(limb_lens[starts[i] : ends[i]]) / pixels_per_cm)
                    cycle_dur_secs.append(cycle_dur_frame / frame_rate)
                    cycle_dur_frames.append(cycle_dur_frame)

                    stride_lens.append(stride_len_cm)

                    step_heights.append(step_height)

                    cycle_vs.append(cycle_v) # (px/s) / (px/cm) = cm/s
                    # or use averaged v from euclidean distance between every two frames? 
                    # (that would consider the entire trajectory of a stride, including vertical movement / lift,
                    # not just the distance between start and end point)

                    stance_dur_secs.append(stance_dur_frame / frame_rate)
                    swing_dur_secs.append(swing_dur_sec)
                    swing_percentages.append(swing_perc)
                    stance_percentages.append(stance_perc)

                    if stance_dur_frame == cycle_dur_frame:
                        max_v_during_swings.append(np.nan)
                    else:
                        max_v_during_swings.append(max(velocities[starts[i]:ends[i]]) / pixels_per_cm)

                    mtp_joint_extension = max(angles_toe_mtp_ankle[starts[i] : ends[i]])
                    ankle_joint_extension = max(angles_mtp_ankle_knee[starts[i] : ends[i]])
                    knee_joint_extension = max(angles_ankle_knee_hip[starts[i] : ends[i]])
                    hip_joint_extension = max(angles_knee_hip_crest[starts[i] : ends[i]])

                    mtp_joint_flexion = min(angles_toe_mtp_ankle[starts[i] : ends[i]])
                    ankle_joint_flexion = min(angles_mtp_ankle_knee[starts[i] : ends[i]])
                    knee_joint_flexion = min(angles_ankle_knee_hip[starts[i] : ends[i]])
                    hip_joint_flexion = min(angles_knee_hip_crest[starts[i] : ends[i]])

                    mtp_joint_amplitude = mtp_joint_extension - mtp_joint_flexion
                    ankle_joint_amplitude = ankle_joint_extension - ankle_joint_flexion
                    knee_joint_amplitude = knee_joint_extension - knee_joint_flexion
                    hip_joint_amplitude = hip_joint_extension - hip_joint_flexion

                    mtp_joint_sds.append(np.std(angles_toe_mtp_ankle[starts[i] : ends[i]]))
                    ankle_joint_sds.append(np.std(angles_mtp_ankle_knee[starts[i] : ends[i]]))
                    knee_joint_sds.append(np.std(angles_ankle_knee_hip[starts[i] : ends[i]]))
                    hip_joint_sds.append(np.std(angles_knee_hip_crest[starts[i] : ends[i]]))

                    mtp_joint_extensions.append(mtp_joint_extension)
                    mtp_joint_flexions.append(mtp_joint_flexion)
                    mtp_joint_amplitudes.append(mtp_joint_amplitude)

                    ankle_joint_extensions.append(ankle_joint_extension)
                    ankle_joint_flexions.append(ankle_joint_flexion)
                    ankle_joint_amplitudes.append(ankle_joint_amplitude)

                    knee_joint_extensions.append(knee_joint_extension)
                    knee_joint_flexions.append(knee_joint_flexion)
                    knee_joint_amplitudes.append(knee_joint_amplitude)

                    hip_joint_extensions.append(hip_joint_extension)
                    hip_joint_flexions.append(hip_joint_flexion)
                    hip_joint_amplitudes.append(hip_joint_amplitude)

                    drag, drag_percent = find_drag(smooth_toe_y, stance_dur_frame, treadmill_y, starts[i], ends[i])
                    drag_ts.append(drag/frame_rate)
                    if swing_dur_sec and stance_dur_frame > 0:
                        drag_percentages.append((drag/frame_rate) / swing_dur_sec)    
                    elif swing_dur_sec and stance_dur_frame == 0:
                        drag_percentages.append(np.nan)
                    else:
                        # no swing phase detected!
                        drag_percentages.append(1)

                else:

                    cycle_dur_secs.append(np.nan)
                    cycle_dur_frames.append(np.nan)
                    stride_lens.append(np.nan)
                    cycle_vs.append(np.nan)
                    stance_dur_secs.append(np.nan)
                    swing_dur_secs.append(np.nan)
                    swing_percentages.append(np.nan)
                    stance_percentages.append(np.nan)
                    limb_len_means.append(np.nan)
                    limb_len_maxs.append(np.nan)
                    limb_len_mins.append(np.nan)
                    limb_len_sds.append(np.nan)
                    step_heights.append(np.nan) 
                    max_v_during_swings.append(np.nan)
                    mtp_joint_extensions.append(np.nan)
                    mtp_joint_flexions.append(np.nan)
                    mtp_joint_amplitudes.append(np.nan)
                    ankle_joint_extensions.append(np.nan)
                    ankle_joint_flexions.append(np.nan)
                    ankle_joint_amplitudes.append(np.nan)
                    knee_joint_extensions.append(np.nan)
                    knee_joint_flexions.append(np.nan)
                    knee_joint_amplitudes.append(np.nan)
                    hip_joint_extensions.append(np.nan)
                    hip_joint_flexions.append(np.nan)
                    hip_joint_amplitudes.append(np.nan)
                    mtp_joint_sds.append(np.nan)
                    ankle_joint_sds.append(np.nan)
                    knee_joint_sds.append(np.nan)
                    hip_joint_sds.append(np.nan)
                    drag_ts.append(np.nan)
                    drag_percentages.append(np.nan)

            starts_included = np.array(starts_included)
            ends_included = np.array(ends_included)
        
        for i in range(len(starts)):
            
            if starts[i] not in starts_included:
                
                dtw_x_plane_5_means.append(np.nan)
                dtw_x_plane_5_sds.append(np.nan)
                dtw_y_plane_5_means.append(np.nan)
                dtw_y_plane_5_sds.append(np.nan)
                dtw_xy_plane_5_means.append(np.nan)
                dtw_xy_plane_5_sds.append(np.nan)
                dtw_x_plane_10_means.append(np.nan)
                dtw_x_plane_10_sds.append(np.nan)
                dtw_y_plane_10_means.append(np.nan)
                dtw_y_plane_10_sds.append(np.nan)
                dtw_xy_plane_10_means.append(np.nan)
                dtw_xy_plane_10_sds.append(np.nan)
                
            else:
                included_index = int(np.where(starts[i] == starts_included)[0])                
                
                if included_index <= len(starts_included)-5:
                    
                    toe_xs = parse_stride_lists(smooth_toe_x, starts_included, ends_included, included_index, 5, True)
                    pairwise_dtw_res_x_5 = pairwise_dtw(toe_xs)
                    dtw_x_plane_5_mean = np.mean(pairwise_dtw_res_x_5)
                    dtw_x_plane_5_sd = np.std(pairwise_dtw_res_x_5)

                    toe_ys = parse_stride_lists(smooth_toe_y, starts_included, ends_included, included_index, 5, True)
                    pairwise_dtw_res_y_5 = pairwise_dtw(toe_ys)
                    dtw_y_plane_5_mean = np.mean(pairwise_dtw_res_y_5)
                    dtw_y_plane_5_sd = np.std(pairwise_dtw_res_y_5)

                    toe_xys = [np.append(toe_xs[j], toe_ys[j]).reshape(2,-1).T for j in range(5)]
                    pairwise_dtw_res_xy_5 = pairwise_dtw(toe_xys)
                    dtw_xy_plane_5_mean = np.mean(pairwise_dtw_res_xy_5)
                    dtw_xy_plane_5_sd = np.std(pairwise_dtw_res_xy_5)

                    if included_index <= len(starts_included)-10:
                        toe_xs = parse_stride_lists(smooth_toe_x, starts_included, ends_included, included_index, 10, True)
                        pairwise_dtw_res_x_10 = pairwise_dtw(toe_xs)
                        dtw_x_plane_10_mean = np.mean(pairwise_dtw_res_x_10)
                        dtw_x_plane_10_sd = np.std(pairwise_dtw_res_x_10)

                        toe_ys = parse_stride_lists(smooth_toe_y, starts_included, ends_included, included_index, 10, True)
                        pairwise_dtw_res_y_10 = pairwise_dtw(toe_ys)
                        dtw_y_plane_10_mean = np.mean(pairwise_dtw_res_y_10)
                        dtw_y_plane_10_sd = np.std(pairwise_dtw_res_y_10)

                        toe_xys = [np.append(toe_xs[j], toe_ys[j]).reshape(2,-1).T for j in range(10)]
                        pairwise_dtw_res_xy_10 = pairwise_dtw(toe_xys)
                        dtw_xy_plane_10_mean = np.mean(pairwise_dtw_res_xy_10)
                        dtw_xy_plane_10_sd = np.std(pairwise_dtw_res_xy_10)

                        dtw_x_plane_10_means.append(dtw_x_plane_10_mean)
                        dtw_x_plane_10_sds.append(dtw_x_plane_10_sd)
                        dtw_y_plane_10_means.append(dtw_y_plane_10_mean)
                        dtw_y_plane_10_sds.append(dtw_y_plane_10_sd)
                        dtw_xy_plane_10_means.append(dtw_xy_plane_10_mean)
                        dtw_xy_plane_10_sds.append(dtw_xy_plane_10_sd)
                    else:
                        dtw_x_plane_10_means.append(np.nan)
                        dtw_x_plane_10_sds.append(np.nan)
                        dtw_y_plane_10_means.append(np.nan)
                        dtw_y_plane_10_sds.append(np.nan)
                        dtw_xy_plane_10_means.append(np.nan)
                        dtw_xy_plane_10_sds.append(np.nan)

                    dtw_x_plane_5_means.append(dtw_x_plane_5_mean)
                    dtw_x_plane_5_sds.append(dtw_x_plane_5_sd)
                    dtw_y_plane_5_means.append(dtw_y_plane_5_mean)
                    dtw_y_plane_5_sds.append(dtw_y_plane_5_sd)
                    dtw_xy_plane_5_means.append(dtw_xy_plane_5_mean)
                    dtw_xy_plane_5_sds.append(dtw_xy_plane_5_sd)
                    
                else:
                    
                    dtw_x_plane_5_means.append(np.nan)
                    dtw_x_plane_5_sds.append(np.nan)
                    dtw_y_plane_5_means.append(np.nan)
                    dtw_y_plane_5_sds.append(np.nan)
                    dtw_xy_plane_5_means.append(np.nan)
                    dtw_xy_plane_5_sds.append(np.nan)
                    dtw_x_plane_10_means.append(np.nan)
                    dtw_x_plane_10_sds.append(np.nan)
                    dtw_y_plane_10_means.append(np.nan)
                    dtw_y_plane_10_sds.append(np.nan)
                    dtw_xy_plane_10_means.append(np.nan)
                    dtw_xy_plane_10_sds.append(np.nan)
                
#     for bodypart in bodyparts:
#         for coord in ['x', 'y']:
#             pd_dataframe[f'{bodypart} {coord}'] = butterworth_filter(pd_dataframe[f'{bodypart} {coord}'], frame_rate, cutoff_f)
        
    
    

    return pd.DataFrame(data=np.array([
                            limbs,
                            starts_all,
                            ends_all,
                            cycle_dur_secs,
                            cycle_dur_frames,
                            cycle_vs,
                            stride_lens, 
                            stance_dur_secs, 
                            swing_dur_secs,
                            swing_percentages, 
                            stance_percentages, 
                            limb_len_means,
                            limb_len_maxs,
                            limb_len_mins,
                            limb_len_sds,
                            step_heights,
                            max_v_during_swings,
                            mtp_joint_extensions,
                            mtp_joint_flexions,
                            mtp_joint_amplitudes,
                            mtp_joint_sds,
                            ankle_joint_extensions,
                            ankle_joint_flexions,
                            ankle_joint_amplitudes,
                            ankle_joint_sds,
                            knee_joint_extensions,
                            knee_joint_flexions,
                            knee_joint_amplitudes,
                            knee_joint_sds,
                            hip_joint_extensions,
                            hip_joint_flexions,
                            hip_joint_amplitudes,   
                            hip_joint_sds,
                            drag_ts, 
                            drag_percentages,
                            dtw_x_plane_5_means,
                            dtw_x_plane_5_sds,
                            dtw_y_plane_5_means,
                            dtw_y_plane_5_sds,
                            dtw_xy_plane_5_means,
                            dtw_xy_plane_5_sds,
                            dtw_x_plane_10_means,
                            dtw_x_plane_10_sds,
                            dtw_y_plane_10_means,
                            dtw_y_plane_10_sds,
                            dtw_xy_plane_10_means,
                            dtw_xy_plane_10_sds        
                            ]).T, 
                            columns=[
                                'limb (hind left / right)',
                                'stride_start (frame)',
                                'stride_end (frame)',
                                'cycle duration (s)',
                                'cycle duration (no. frames)',
                                'cycle velocity (cm/s)',
                                'stride length (cm)', 
                                'stance duration (s)', 
                                'swing duration (s)',
                                'swing percentage (%)', 
                                'stance percentage (%)', 
                                'mean toe-to-crest distance (cm)',
                                'max toe-to-crest distance (cm)',
                                'min toe-to-crest distance (cm)',
                                'toe-to-crest distance SD (cm)',
                                'step height (cm)',
                                'max velocity during swing (cm/s)',
                                'mtp joint extension (deg)',
                                'mtp joint flexion (deg)',
                                'mtp joint amplitude (deg)',
                                'mtp joint SD (deg)',
                                'ankle joint extension (deg)',
                                'ankle joint flexion (deg)',
                                'ankle joint amplitude (deg)',
                                'ankle joint SD (deg)',
                                'knee joint extension (deg)',
                                'knee joint flexion (deg)',
                                'knee joint amplitude (deg)',
                                'knee joint SD (deg)',
                                'hip joint extension (deg)',
                                'hip joint flexion (deg)',
                                'hip joint amplitude (deg)', 
                                'hip joint SD (deg)', 
                                'drag duration (s)', 
                                'drag percentage (%)',
                                'DTW distance x plane 5 strides mean',
                                'DTW distance x plane 5 strides SD',
                                'DTW distance y plane 5 strides mean',
                                'DTW distance y plane 5 strides SD',
                                'DTW distance xy plane 5 strides mean',
                                'DTW distance xy plane 5 strides SD',
                                'DTW distance x plane 10 strides mean',
                                'DTW distance x plane 10 strides SD',
                                'DTW distance y plane 10 strides mean',
                                'DTW distance y plane 10 strides SD',
                                'DTW distance xy plane 10 strides mean',
                                'DTW distance xy plane 10 strides SD',
                                ]), pd_dataframe, is_stance, bodyparts