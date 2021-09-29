import cv2

def face_detect(img_path, classifier, text):
    """
    return image with detected face
    """
    #open img
    img_face = cv2.imread(img_path)
    gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY) #convert img to gray format

    #detect face
    faces = classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    #draw bounding box and write text
    for (x, y, w, h) in faces:
        cv2.rectangle(img_face, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_face, text, (x-1, y-3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

    return img_face

if __name__ == '__main__':
    img_path = 'face.png'
    classifier_path = 'haarcascade_frontalface_default.xml'

    # open the haar cascade
    faceCascade = cv2.CascadeClassifier(classifier_path)
    
    #detect face
    img_face = face_detect(img_path, faceCascade, 'detected face')

    cv2.imwrite('face_det.png', img_face)

