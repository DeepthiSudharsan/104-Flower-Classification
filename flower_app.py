import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import PIL
import tensorflow as tf
import numpy as np
import os
import cv2

import urllib.request
urllib.request.urlretrieve("https://drive.google.com/uc?export=download&id=19DI-ONYkg56f2ZjUAkWf55mD_jsW4e7A", "MobileNet_model.hdf5")

#Eff_model = tf.keras.models.load_model('./eff_model.hdf5') 
Eff_model = tf.keras.models.load_model('./MobileNet_model.hdf5') 
uploaded_file = st.file_uploader("Choose a file")

def validate_set(img):
    X_valid = []
    image = Image.open(img)
    #image = ImageOps.grayscale(image)
        
    image = np.array(image)
    image_data_as_arr = np.asarray(image)
        
    X_valid.append(image_data_as_arr)
    X_valid = np.asarray(X_valid)   
    X_valid = tf.expand_dims(X_valid, axis=-1)
    return X_valid

lanel_dic = {"toad lily" :  38, "love in the mist":  61, "monkshood":  75, "azalea":  54, "fritillary":  6, "silverbush":  17, "canterbury bells":  8, "stemless gentian":  59, "pink primrose":  103, "buttercup":  62, "poinsettia":  92, "desert-rose":  76, "bird of paradise":  28, "columbine":  16, "cyclamen":  83, "frangipani":  93, "sweet pea":  19, "siam tulip":  26, "great masterwort":  89, "hard-leaved pocket orchid":  22, "marigold":  53, "foxglove":  57, "wild pansy":  9, "windflower":  84, "daisy":  64, "tiger lily":  18, "purple coneflower":  23, "orange dahlia":  41, "globe-flower":  43, "lilac hibiscus":  85, "fire lily":  3, "balloon flower":  87, "iris":  101, "bishop of llandaff":  71, "yellow iris":  51, "garden phlox":  0, "alpine sea holly":  21, "geranium":  60, "pink quill":  35, "tree poppy":  44, "spear thistle":  69, "bromelia":  82, "common dandelion":  50, "sword lily":  97, "peruvian lily":  91, "carnation":  96, "cosmos":  46, "spring crocus":  25, "lotus":  94, "bolero deep blue":  74, "anthurium":  79, "rose":  96, "water lily":  32, "primula":  5, "blackberry lily":  70, "gaura":  95, "trumpet creeper":  52, "globe thistle":  7, "sweet william":  40, "hippeastrum":  15, "snapdragon":  47, "mexican petunia":  49, "petunia":  15, "gazania":  10, "king protea":  11, "blanket flower":  34, "common tulip":  102, "giant white arum lily":  65, "wild rose":  1, "morning glory":  4, "thorn apple":  98, "pincushion flower":  39, "tree mallow":  13, "canna lily":  91, "camellia":  99, "pink-yellow dahlia":  63, "bee balm":  80, "wild geranium":  24, "artichoke":  38, "black-eyed susan":  58, "ruby-lipped cattleya":  86, "clematis":  55, "prince of wales feathers":  81, "hibiscus":  42, "cautleya spicata":  67, "lenten rose":  36, "red ginger":  14, "colt's foot":  90, "mallow":  31, "californian poppy":  68, "corn poppy":  52, "moon orchid":  45, "passion flower":  48, "grape hyacinth":  78, "japanese anemone":  66, "watercress":  72, "cape flower":  29, "osteospermum":  77, "barberton daisy":  20, "bougainvillea":  27, "magnolia":  100, "sunflower":  90, "daffodil":  12, "wallflower":  56}

if(uploaded_file):
  print(uploaded_file)
  st.header("Image")
  st.image(uploaded_file)
  X_val = validate_set(uploaded_file)
  y_pred = Eff_model.predict(X_val)
  # print(y_pred)
  Y_pred_classes = np.argmax(y_pred,axis=1)
  # print(Y_pred_classes)
  # print(np.argpartition(y_pred[0], -4)[-4:])
  keys = [k for k, v in lanel_dic.items() if v == Y_pred_classes]
  st.write(keys[0])



