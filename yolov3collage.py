# http://olivain.art

from PIL import Image, ImageFont, ImageDraw, ImageTk
import cv2
import numpy as np
import os

currentPath = os.getcwd()+"/" # chemin actuel dans lequel se situe le programme
config_path = currentPath+"yolov3.cfg" # organisation du reseau de neurones
weights_path = currentPath+"yolov3.weights" # hierarchie du reseau de neurones
labels = open(currentPath+"coco.names").read().strip().split("\n") # noms des objets detectables
fontFileName=currentPath+'RobotoSlab-Regular.ttf' # police pour affichage sur l'image
fontSize = 25 # taille de la police
loadedFont = ImageFont.truetype(fontFileName,fontSize) # chargement de la police pour affichage dans la fenetre
blurRate = 25 # taux de flou
CONFIDENCE = 0.5 #taux de confiance (50%)
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
boxes, confidences, class_ids = [], [], [] # tableaux utiles pour la detection

imgTarget = "./imgs/test.png" # image a soumettre a l'algorithme
image = cv2.imread(imgTarget) # ouverture de l'image
origh, origw = image.shape[:2] #taille originale de l'image
net = cv2.dnn.readNetFromDarknet(config_path, weights_path) # initialisation du reseau de neurones
# tranformation de l'image reduite en "blob" pur traitement par le reseau de neurones :
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
image.shape: (1200, 1800, 3) #?
blob.shape: (1, 3, 416, 416) #?
# on envoie le blob au reseau de neurones
net.setInput(blob)
#obtention des differentes couches du reseau
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] # /!\ Cette ligne echoue avec openCV 4.5.4 mais pas avec openCV 4.5.3 !!!!!
layer_outputs = net.forward(ln) # on effectue la detection d'objets
for output in layer_outputs: #pour chaque valeurs retournee par les couches du reseau de neurones..
    for detection in output: #pour chaque detection dans chaque valeur..
        scores = detection[5:] # informations sur la detection
        class_id = np.argmax(scores) #extraction de l'ID detecte (quel objet?)
        confidence = scores[class_id] #extraction du taux de confiance dans la detection effectuee
        if confidence > CONFIDENCE: # si le taux de confiance est superieur au niveau minimum defini (50%)
            box = detection[:4] * np.array([origw, origh, origw, origh])  # obtention des coordoonees
            (centerX, centerY, width, height) = box.astype("int") ## obtention des coordoonees du centre de l'objet detecte et de sa taille
            coord_x = int(centerX - (width / 2)) # coordonnee x du point haut gauche
            coord_y = int(centerY - (height / 2)) # coordonnee y du point haut gauche
            boxes.append([coord_x, coord_y, int(width), int(height)]) # ajout des coordoonees et taille de l'objet a la liste des objets
            confidences.append(float(confidence)) # ajout du taux de confiance pour cet objet a a liste des taux de confiance
            class_ids.append(class_id) # ajout de l'id de l'objet detecte a la liste des IDs des objets detectes

# la fonction suivante supprime d'eventuelles detections multiples d'un meme objet
# (cf. https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/ ) :
idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
#si nous avons au moins un objet detecte
if len(idxs) > 0:
    # Iterations dans les donnees recuperees (detections) pour dessin et floutage de l'image
    for i in idxs.flatten():
        x, y = boxes[i][0], boxes[i][1] # recuperation des coordonnees de l'objet detecte
        w, h = boxes[i][2], boxes[i][3] # recuperation de la hauteur et de la largeur de l'objet detecte
        #correction necessaire si les valeurs sont negatives:
        if (x < 0): x = 0
        if (y < 0): y = 0
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(0,0,0), thickness=1) # on dessine un rectangle dans l'image a l'endroit de la detection
        text= f"{labels[class_ids[i]]}" # on prepare le texte (nom de l'objet detecte) pour dessin sur l'image
        image[y+1:y+h, x+1:x+w] = cv2.blur(image[y+1:y+h, x+1:x+w], (blurRate, blurRate)) # on floute le rectangle de la detection
        # pour ecrire le texte (label de l'objet detecte) sur les images et avec une police de notre choix
        # nous devons passer par PIL (la police est chargee, pour l'image reduite, en variable globale au debut du programme (loadedFont))
        im_p = Image.fromarray(image) #trasfert de l'image reduite depuis OpenCV (numpy) vers Pil
        draw = ImageDraw.Draw(im_p) #ouverture de l'image reduite pour dessin
        draw.text((x+2,y-2),text,(0,0,0),font=loadedFont) # pose du texte sur l'image reduite
        image = np.array(im_p)# conversion depuis Pil vers OpenCV (numpy)

    cv2.imshow("image",image) # affichage de l'image finale
    cv2.waitKey(0)
