import cv2
import os


def VideoFaceDet(window_name, classifier, camera_idx, fps, size, catch_frame_num=None):
    """
    Show every single frame processing and save detected video
    Args:
        window_name -> str: the name of showing window
        classifier -> cv2.CascadeClassifier: face detector
        camera_idx -> str / int: the input for VideoCapture, file path or int for camera
        fps -> int: fps of saved vedio
        size -> tuple(int, int): size of video
        catch_frame_num -> default[None] / int: None to save all frames or save limit num
    """
    cv2.namedWindow(window_name)

    file_path = './face_det.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4v / DIVX
    # #video writer
    videoWriter = cv2.VideoWriter(file_path,fourcc, fps, size) # aegs[-2:] = fps, size

    # capture video
    cap = cv2.VideoCapture(camera_idx)
    if catch_frame_num is None:
        catch_frame_num = cap.get(7)
 
    color = (0, 255, 0)

    num = 0
    while cap.isOpened():
        status, frame = cap.read() # read single frame
        if not status:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to gray format
         # save frame to tmp path
        img_path = "tmp.jpg"
        faceDets = classifier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceDets) > 0:
            for faceDet in faceDets:  
                x, y, w, h = faceDet
                # draw bounding box
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

        num += 1
        if num > (catch_frame_num): break

        cv2.imwrite(img_path, frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

        #write img to video
        img = cv2.imread(img_path)
        img = cv2.resize(img, size)
        videoWriter.write(img)

        #show img in window
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    
    # cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # open the haar cascade
    facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    VideoFaceDet("get face", facecascade, 'face4.mp4', 30, (1440,1080), 450)