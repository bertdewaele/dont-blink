""" Face Detection, Bert Dewaele """

import cv2  

def detect_faces(frame, cascade):
    """ Detect faces in frame, given a face haarcascade """
    rects = cascade.detectMultiScale(frame, 1.2, 4, 
            cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))

    if len(rects) == 0:
        return [], frame
    
    rects[:, 2:] += rects [:, :2]
    return rects, frame

def detect_eyes(frame, cascade):
    """ Detect eyes in frame, given a eye haarcascade """
    rects = cascade.detectMultiScale(frame, 1.1, 2, 
            cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))
    
    if len(rects) == 0:
        return [], frame
    
    rects[:, 2:] += rects [:, :2]
    return rects, frame


def box_image(rects, frame):
    """ Draw rectangles on frame """

    for x1, y1, x2, y2 in rects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

if __name__ == '__main__':
    
    CAM = 'Webcam Stream'
    ESC = 27
    
    # setup window and capture
    cv2.namedWindow(CAM)
    cap = cv2.VideoCapture(0)

    # setup classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    if cap.isOpened(): 
        rval, frame = cap.read()
    else:
        rval = False
        print()
    
    while rval:
        cv2.imshow(CAM, frame)
        rval, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_rects, frame = detect_faces(frame, face_cascade)
        eye_rects, frame = detect_eyes(frame, eye_cascade)
        box_faces(face_rects, frame)
        box_faces(eye_rects, frame)

        key = cv2.waitKey(10)
        if key == ESC: # exit on ESC
            break

    cv2.destroyWindow(CAM)
