#Seila Becirovic 1255/17118
#POOS Z5


#Rjesenje implemntirano na ovaj nacin moze vrsiti treniranje i predikciju za vise subjekata
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

#Crtanje kvadrata
def draw_rec(img, rec):
    (x, y, w, h) = rec
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 1)

#Ispisivanje teksta
def draw_text(img, t, x, y):
    cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),2)

#Kreira sve moguce putanje
def create_paths(training_data_dir_path):
    subdir_names = os.listdir(training_data_dir_path)
    subdir_paths = [training_data_dir_path + "/" + subdir_name for subdir_name in subdir_names]
    #Pokupe se nazivi svih subjekata za koje se vrsi treniranje - te se prebace u skup radi dobivanja jedinstvenih
    #Nakon toga se skup prebaci u listu radi mogucnosti indeksiranja
    subjects = [subdir_name for subdir_name in subdir_names]
    subjects = list(set(subjects))
    images_paths = []
    for subdir_path in subdir_paths:
        images_names = os.listdir(subdir_path)
        images_paths += [subdir_path + "/" + img for img in images_names if not img.startswith(".")]
    return subjects, images_paths

#Detektuje lice primjenom klasifikatora lbpcascade_frontalface radi brzine
#Dovoljno je promijeniti klasifikator na haarcascade_frontalface_default, u cilju postizanje evenutalno vece tacnosti
def face_detect(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.2, minNeighbors = 5)
    if len(faces) == 0:
        return None,None
    (x,y,w,h) = faces[0]
    return gray_img[y:y+w, x:x+h], faces[0]

#Priprema testne podatke
def prepare_data(subjects, images_paths):
    labels = []
    faces = []
    rectangles = []
    counter = 1;
    for image_path in images_paths:
        image = cv2.imread(image_path)
        face, rectangle = face_detect(image)
        if face is not None:
            faces.append(face)
            rectangles.append(rectangle)
            for i in range(len(subjects)):
                if subjects[i] in image_path:
                    #Labele moraju biti integeri
                    labels.append(i+1)
                    break
    return faces, labels, rectangles

#Treniranje opencv klasifikatora
def train_data(faces,labels):
    #Izmjena predlozenog - nova verzija face metoda
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))
    return face_recognizer

#Funkcija za predvidjanje
def predict(test_img_path,subjects):
    test_img = cv2.imread(test_img_path)
    img = test_img.copy()
    face, rectangle = face_detect(test_img)
    label, confidence = face_recognizer.predict(face)
    label = subjects[label-1]
    draw_rec(img,rectangle)
    label_text = label + ", confidence " + str(confidence)
    draw_text(img, label_text, rectangle[0], rectangle[1]-5)
    print(test_img_path + ": " + label_text)
    return img

#Funkcija za testiranje koristenjem putanja
def test(test_img_paths,subjects):
    for image_path in test_img_paths:
        predicted_img = predict(image_path,subjects)
        cv2.imshow(image_path, predicted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#Potrebno je proslijedjivati subjekte radi ispisivanja naziva
#Moze se kreirati lista subjekata koja je globalna (nema potrebe za proslijedjivanjem dodatnog parametra)
print("Kreiranje putanja")
subjects, images_paths= create_paths("training-data")
print("Priprema podataka")
faces, labels, rectangles = prepare_data(subjects, images_paths)
print("Treniranje")
face_recognizer = train_data(faces, labels)
#Testiranje se vrsi na preuzetim slikama u folderu test-data
print("Testiranje")
s,images_paths= create_paths("test-data")
test(images_paths,subjects)
