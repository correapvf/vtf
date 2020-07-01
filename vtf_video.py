import numpy as np
import cv2
import vtf_config as c
import vtf_functions as f

def extract_frames(file, folder, start, duration, start_real, last_keyframe):
    # open ffmpeg to capture frames
    cap = f.Capture(file, start, duration, start_real)
    fps = int(cap.fps)
    center_img = f.CenterImg(cap)
    if c.display:
        plot = f.Plot(fps)
    
    # variables to control the main loop
    keyframe = last_keyframe
    new_keyframe = True
    time_overaly = 0
    time_blured = 0
    counter_last_keyframe = 0
    ed = None
    is_forced = False
    old_keyframe = np.zeros((c.out_H, c.out_W), np.uint8)

    elapsed_max = int(c.elapsed_max * cap.fps) # thr to always save the frame
    
    # store variables for csv
    # log.csv
    frame_log = []; frame_app = frame_log.append
    time_log = []; time_app = time_log.append
    method_log = []; method_app = method_log.append
    elapsed_log = []; elapsed_app = elapsed_log.append

    #output.csv
    filename_log = []; filename_app = filename_log.append
    time_real_log = []; time_real_app = time_real_log.append

    mask = f.create_mask() # mask for Shi-Tomasi points detector

    # thr for farneback flow
    x_flow_thr = c.out_W * (1 - c.overlap)
    y_flow_thr = c.out_H * (1 - c.overlap)

    # variables used when downsize frame for farneback flow calculation
    xd = c.out_W // c.downsize_flow
    yd = c.out_H // c.downsize_flow
    n_percent = int(xd*yd*c.percent_flow)
    xd = (c.out_W - xd) // 2
    yd = (c.out_H - yd) // 2

    while cap.ret():
        # load frame
        img = cap.retrieve()

        # MODIFY function check_overlay in function.py as needed
        if c.check_overlay(img): # if it detects an overlay
            if c.display:
                cv2.putText(img, 'Frame descaterted by overlay', **c.text_params)
                cv2.imshow('original frame', cv2.resize(img, **c.resize_params))
                k = cv2.waitKey(1)
                if k == 27:
                    print(f'Interrupted at {cap.time()}')
                    break
            time_overaly += 1
            continue # skip to next frame

        if time_overaly > 0:
            frame_app('overlay')
            time_app(cap.time())
            method_app('overlay')
            elapsed_app(time_overaly*cap.t_add)

            new_keyframe = True
            time_overaly = 0
            ed = None
            counter_last_keyframe = 0
        
        # convert to single channel
        gray = img[c.overlay_top:,:,1] # use only green channel

        # laplace filter
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_16S, ksize=5))
        blur_detect = lap.max()
        
        # check if elapsed frames from last keyframe is above thr
        if counter_last_keyframe > elapsed_max:
            new_keyframe = True
            is_forced = True
            time_blured = 0
            ed = None    
        else:
            # check if frame is too burred
            if blur_detect < c.blur_thr:
                if c.display:   
                    f.skip_n_display(plot, cap, img, blur_detect)
                else:
                    cap.skip() # skip the frames without processing them - much faster
                time_blured += 1
                counter_last_keyframe += c.blur_skip
                continue
            
            # too much consecutive frames bluried
            if time_blured > 2:
                frame_app('blur')
                time_app(cap.time())
                method_app('blur')
                elapsed_app(time_blured*c.blur_skip*cap.t_add)

                new_keyframe = True
                time_blured = 0
                ed = None
                counter_last_keyframe = 0


        if new_keyframe:
            # CREATE A NEW KEYFRAME
            ed = center_img.process(gray, ed) # define the center 
            roi = gray[ed[0]:ed[1], ed[2]:ed[3]] # cut image in the roi
            norm = f.normalize(roi) # normalize image contrast and brightness

            # try to generate new points
            p0 = cv2.goodFeaturesToTrack(norm, mask = mask, **c.feature_params)
            npoints = len(p0)

            if npoints < c.points_to_LK:
                # then dense flow
                u_sum = 0
                v_sum = 0
                LK = False
                method = 'FR'
                prvs = norm[yd:(c.out_H-yd), xd:(c.out_W-xd)]
            else:
                LK = True
                method = 'LK'
                prvs = norm.copy()
            
            if is_forced:
                is_forced = False
                method = 'force'

            if c.display:
                u = 0; v = 0
                center_point = (ed[2] + center_img.x_crop, ed[0] + center_img.x_crop)
                good_new = p0
                cv2.imshow('current', cv2.resize(norm, **c.resize_params))
                cv2.imshow('prvs', cv2.resize(old_keyframe, **c.resize_params))
                old_keyframe = norm.copy()
            
            # SAVE FRAME AND APPEND DATA
            output_frame = f'frame{keyframe:03d}.jpg'
            time_formated = cap.time()

            cv2.imwrite(f.concat_path(folder, output_frame), frame_format[c.save_format](roi, norm, img, ed))

            filename_app(output_frame)
            time_real_app(cap.real_time())

            frame_app(output_frame)
            time_app(time_formated)
            method_app(method)
            t_elapsed = counter_last_keyframe*cap.t_add
            elapsed_app(f'{t_elapsed:03.1f}')

            remaining = duration - (cap.counter*cap.t_add)
            print(f'{keyframe:03d} #keyframe at {time_formated}. Remaining {remaining:.0f}s')
            new_keyframe = False
            keyframe += 1
            counter_last_keyframe = 0

 
        else:
            # CALCULATE FLOW 
            counter_last_keyframe += 1
            roi = gray[ed[0]:ed[1], ed[2]:ed[3]] # cut image in the roi
            norm = f.normalize(roi) 

            if LK:
                # by using Lucas-Kanade
                if time_blured:
                    time_blured = 0
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, norm, p0, None, **c.lk_params_blur)
                else:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, norm, p0, None, **c.lk_params)
            
                # Select good points
                index = st == 1
                err[~index] = 0
                index = np.logical_and(index, err > 1.0)
                good_new = p1[index]
                npoints = len(good_new)
            
                # if too few points were left, them create new keyframe
                if npoints > 2:
                    p0 = good_new.reshape(-1,1,2)
                    prvs = norm.copy()
                else:
                    new_keyframe = True

            
            else:
                # by using Farneback
                current = norm[yd:(c.out_H-yd), xd:(c.out_W-xd)]
                if time_blured:
                    time_blured = 0
                    flow = cv2.calcOpticalFlowFarneback(prvs, current, None, **c.farneback_params_blur)
                else:
                    flow = cv2.calcOpticalFlowFarneback(prvs, current, None, **c.farneback_params)

                flow_rectify = f.sobel_cor(lap.mean())
                u = f.mean_interval(flow[...,0], n_percent)*flow_rectify
                v = f.mean_interval(flow[...,1], n_percent)*flow_rectify
                u_sum += u
                v_sum += v

                if abs(u_sum) > x_flow_thr or abs(v_sum) > y_flow_thr:
                    new_keyframe = True
                else:
                    prvs = current.copy()

        if c.display:
            # display main window and update plot
            if not cap.counter % fps:
                time_real = cap.real_time()
            
            plot.update(blur_detect)

            cv2.rectangle(img, (ed[2], ed[0]+c.overlay_top), (ed[3], ed[1]+c.overlay_top), (0,0,255), 3)
            cv2.putText(img, f'time: {time_real}, keyframe: {keyframe:03d}', **c.text_params)
            
            if LK:
                for new in good_new.astype(int):
                    x, y = new.ravel()
                    x += ed[2]
                    y += ed[0] + c.overlay_top
                    cv2.circle(img, (x,y), 3, (0,0,0), -1)
                cv2.putText(img, f'npoints: {npoints:03d}', **c.text_params_sub)
            else:
                end_point = (-int(u*8) + center_point[0], -int(v*8) + center_point[1])
                cv2.arrowedLine(img, center_point, end_point, (0,255,0), 2, tipLength = 0.2)
                cv2.putText(img, f'x:{u_sum:05.1f}, y:{v_sum:05.1f}', **c.text_params_sub)
            
            cv2.imshow('original frame', cv2.resize(img, **c.resize_params))

            k = cv2.waitKey(1)
            if k == 27:
                print(f'Interrupted at {cap.time()}')
                break
    
    # release and destroy all windows
    cap.pipe.stdout.close()
    if c.display:
        cv2.destroyAllWindows()
        plot.close()

    log = {'keyframe': frame_log, 'time': time_log, 'method': method_log, 'duration': elapsed_log}
    return log, filename_log, time_real_log

# define functions to save frame
def sv_gray_norm(roi, norm, img, ed):
    return norm

def sv_gray(roi, norm, img, ed):
    return roi

def sv_color(roi, norm, img, ed):
    return img[(c.overlay_top+ed[0]):(c.overlay_top+ed[1]), ed[2]:ed[3],:]

def sv_color_all(roi, norm, img, ed):
    return img[c.overlay_top:,:,:]

frame_format = {
    'gray_norm': sv_gray_norm,
    'gray': sv_gray,
    'color': sv_color,
    'color_all': sv_color_all
}
