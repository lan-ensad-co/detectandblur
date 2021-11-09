# detectandblur
Détection d'objets dans une image avec Opencv+Yolov3 et floutage de la zone concernée


![example](http://olivain.art/blurd/Vnb4YahtVZVu.jpg)

# installation
```
# /!\ Le script ne fonctionne pas avec opencv-4.5.4 !! /!\
python3 -m pip install opencv-python==4.5.3

#librairies
python3 -m pip install pil
python3 -m pip install numpy

#téléchargement du réseau de neurones pré-entrainé yolov3
wget https://olivain.art/wannabeblog/misc/coco.names
wget https://olivain.art/wannabeblog/misc/yolov3.cfg
wget https://olivain.art/wannabeblog/misc/yolov3.weights

#téléchargement de la police de caractères Roboto Slab utilisée dans le script
curl -O -J -L https://www.fontsquirrel.com/fonts/download/roboto-slab
unzip roboto-slab.zip 

# c'est parti
python3 yolov3collage.py

```
