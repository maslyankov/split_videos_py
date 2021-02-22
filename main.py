# Python 2/3 compatibility
from __future__ import print_function
# Allows use of print like a function in Python 2.x

# Import OpenCV and Numpy modules
import numpy as np
import cv2
from os import path

def crop_video(video_in, x, y, height, width, suffix=None):
    # (x, y, w, h) = cv2.boundingRect(c)
    # cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 20)
    # roi = frame[y:y+h, x:x+w]
    cap = cv2.VideoCapture(video_in)
    # get fps info from file CV_CAP_PROP_FPS, if possible
    fps = int(round(cap.get(5)))
    # check if we got a value, otherwise use any number - you might need to change this
    if fps == 0:
        fps = 30  # so change this number if cropped video has stange steed, higher number gives slower speed

    vid_name = path.basename(video_in)
    vid_name_no_ext = vid_name.split(".")[0]
    suffix = f"_{suffix}" if suffix is not None else ""

    out_cropped = f"{vid_name_no_ext}_cropped" + suffix
    print(f"cropping {vid_name} to {out_cropped}")

    out_path = f'{path.dirname(video_in)}/{out_cropped}.mp4'
    print(f"Is file: {path.isfile(out_path)}")

    suff = 1

    check_path = out_path

    while path.isfile(check_path):
        print(f"Checking {check_path}")

        check_path = f"{out_path.split('.')[0]}_{suff}.mp4"
        suff += 1

    out_path = check_path
    print(f'Saving to {out_path}')

    # output_movie = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))
    output_movie = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        # (height, width) = frame.shape[:2]
        if frame is not None:
            # Crop frame
            cropped = frame[x:x + height, y:y + width]

            # Save to file
            output_movie.write(cropped)

            # Display the resulting frame - trying to move window, but does not always work
            cv2.namedWindow('producing video', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('producing video', cropped.shape[1], cropped.shape[0])
            x_pos = round(width / 2) - round(cropped.shape[1] / 2)
            y_pos = round(height / 2) - round(cropped.shape[0] / 2)
            cv2.moveWindow("producing video", x_pos, y_pos)
            cv2.imshow('producing video', cropped)

            # Press Q on keyboard to stop recording early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Close video capture
    cap.release()
    # Closes the video writer.
    output_movie.release()

    # Make sure all windows are closed
    cv2.destroyAllWindows()

    print('Job done!')


def get_video_info(video_in):
    cap = cv2.VideoCapture(video_in)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Width: {w}, height: {h}, fps: {fps}, nframes: {n_frames}")
    return w, h, fps, n_frames


def split_video(video_in):
    w, h, fps, n_frames = get_video_info(video_in)

    x, y = 0, 0
    height = int(h/2)
    width = w

    print(f"height: {height}, width: {width}")

    # (x, y, w, h) = cv2.boundingRect(c)
    # cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 20)
    # roi = frame[y:y+h, x:x+w]
    cap = cv2.VideoCapture(video_in)
    # get fps info from file CV_CAP_PROP_FPS, if possible
    fps = int(round(cap.get(5)))
    # check if we got a value, otherwise use any number - you might need to change this
    if fps == 0:
        fps = 30  # so change this number if cropped video has stange steed, higher number gives slower speed

    vid_name = path.basename(video_in)
    vid_name_no_ext = vid_name.split(".")[0]

    out_cropped = f"{vid_name_no_ext}_cropped"
    print(f"cropping {vid_name} to {out_cropped}")

    out_path = f'{path.dirname(video_in)}/{out_cropped}.mp4'
    print(f"Is file: {path.isfile(out_path)}")

    suff = 1

    check_path = out_path

    while path.isfile(check_path):
        print(f"Checking {check_path}")

        check_path = f"{out_path.split('.')[0]}_{suff}.mp4"
        suff += 1

    out_path0 = f"{check_path.split('.')[0]}_top.mp4"
    out_path1 = f"{check_path.split('.')[0]}_bottom.mp4"
    print(f'Saving to {out_path0}')
    print(f'Saving to {out_path1}')

    # output_movie = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))
    output_movie0 = cv2.VideoWriter(out_path0, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
    output_movie1 = cv2.VideoWriter(out_path1, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        # (height, width) = frame.shape[:2]
        if frame is not None:
            curr_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Crop frame
            cropped0 = frame[x:x + height, y:y + width]
            cropped1 = frame[x+height:x + height * 2, y:y + width]

            # Save to file
            output_movie0.write(cropped0)
            output_movie1.write(cropped1)

            # Display the resulting frame - trying to move window, but does not always work
            cv2.namedWindow('producing video', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('producing video', cropped0.shape[1], cropped0.shape[0])
            x_pos = round(width / 2) - round(cropped0.shape[1] / 2)
            y_pos = round(height / 2) - round(cropped0.shape[0] / 2)
            cv2.moveWindow("producing video", x_pos, y_pos)
            cv2.imshow('producing video', cropped0)


            print(f"Exporting videos... [frame {curr_frame}/{n_frames}]")

            # Press Q on keyboard to stop recording early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Close video capture
    cap.release()
    # Closes the video writer.
    output_movie0.release()
    output_movie1.release()

    # Make sure all windows are closed
    cv2.destroyAllWindows()

    print('Videos split!')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # crop_video("data/1920x2880_normal.mp4", 0, 0, 1440, 1920)
    split_video("data/1920x2880_normal.mp4")