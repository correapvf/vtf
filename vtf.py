import sys
import pandas as pd
from vtf_video import extract_frames
from vtf_functions import create_path, concat_path, time_difference

file = sys.argv[1]

if file.endswith('.csv'):
    # FOR A CSV WITH MULTIPLE VIDEOS
    videos = pd.read_csv(file)

    time_real_check = True if 'start_time' in videos.columns else False
    output_folder = sys.argv[2] if len(sys.argv) >= 3 else 'frames'

    nrows = len(videos.index)
    for row in videos.itertuples():
        file_stem, folder, last_keyframe = create_path(row.file, output_folder)
        time_dif = time_difference(row.start_file_time, row.end_file_time)

        start_real = row.start_time.strftime('%H:%M:%S') if time_real_check else None

        # process video
        print(f'Starting to process file {file_stem} at {row.start_file_time}. {row.Index+1} of {nrows}')
        log, filename_log, time_real_log = extract_frames(row.file, folder, row.start_file_time,
                                                              time_dif, start_real, last_keyframe)
        print()

        # save log.csv
        log_df = pd.DataFrame(log)
        log_df.to_csv(concat_path(folder, 'log.csv'), mode='a')

        # save output.csv
        output_df = pd.DataFrame({'Station': row.station, 'Video': file_stem,
                                  'Frame': filename_log, 'Time': time_real_log})

        if not row.Index:
            output_df.to_csv(concat_path(output_folder, 'output.csv'), index = False, mode = 'a')
        else:
            output_df.to_csv(concat_path(output_folder, 'output.csv'), index = False, mode = 'a', header = False)

else:
    # FOR ONE VIDEO ONLY
    start = sys.argv[2]
    time_dif = time_difference(start, sys.argv[3])

    start_real = sys.argv[4] if len(sys.argv) >= 5 else '0'
    start_real = None if start_real == '0' else start_real
    output_folder = sys.argv[5] if len(sys.argv) >= 6 else 'frames'

    file_stem, folder, last_keyframe = create_path(file, output_folder)
    log, filename_log, time_real_log  = extract_frames(file, folder, start, time_dif, start_real, last_keyframe)

    # save csv
    log_df = pd.DataFrame(log)
    log_df.to_csv(concat_path(folder, 'log.csv'), index = False, mode = 'a')

    output_df = pd.DataFrame({'Video': file_stem, 'Frame': filename_log, 'Time': time_real_log})
    output_df.to_csv(concat_path(output_folder, file_stem + '.csv'), index = False, mode = 'a')
