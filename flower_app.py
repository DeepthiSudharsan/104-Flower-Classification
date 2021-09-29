import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import PIL
import tensorflow as tf
import numpy as np
import urllib.request
import os.path

lanel_dic = {"Toad lily" :  38, "Love in the mist":  61, "Monkshood":  75, "Azalea":  54, "Fritillary":  6, "Silverbush":  17, "Canterbury bells":  8, "Stemless gentian":  59, "Pink primrose":  103, "Buttercup":  62, "Poinsettia":  92, "Desert-rose":  76, "Bird of paradise":  28, "Columbine":  16, "Cyclamen":  83, "Frangipani":  93, "Sweet pea":  19, "Siam tulip":  26, "Great masterwort":  89, "Hard-leaved pocket orchid":  22, "Marigold":  53, "Foxglove":  57, "Wild pansy":  9, "Windflower":  84, "Daisy":  64, "Tiger lily":  18, "Purple coneflower":  23, "Orange dahlia":  41, "Globe-flower":  43, "Lilac hibiscus":  85, "Fire lily":  3, "Balloon flower":  87, "Iris":  101, "Bishop of llandaff":  71, "Yellow iris":  51, "Garden phlox":  0, "Alpine sea holly":  21, "Geranium":  60, "Pink quill":  35, "Tree poppy":  44, "Spear thistle":  69, "Bromelia":  82, "Common dandelion":  50, "Sword lily":  97, "Peruvian lily":  91, "Carnation":  96, "Cosmos":  46, "Spring crocus":  25, "Lotus":  94, "Bolero deep blue":  74, "Anthurium":  79, "Rose":  96, "Water lily":  32, "Primula":  5, "Blackberry lily":  70, "Gaura":  95, "Trumpet creeper":  52, "Globe thistle":  7, "Sweet william":  40, "Hippeastrum":  15, "Snapdragon":  47, "Mexican petunia":  49, "Petunia":  15, "Gazania":  10, "King protea":  11, "Blanket flower":  34, "Common tulip":  102, "Giant white arum lily":  65, "Wild rose":  1, "Morning glory":  4, "Thorn apple":  98, "Pincushion flower":  39, "Tree mallow":  13, "Canna lily":  91, "Camellia":  99, "Pink-yellow dahlia":  63, "Bee balm":  80, "Wild geranium":  24, "Artichoke":  38, "Black-eyed susan":  58, "Ruby-lipped cattleya":  86, "Clematis":  55, "Prince of wales feathers":  81, "Hibiscus":  42, "Cautleya spicata":  67, "Lenten rose":  36, "Red ginger":  14, "Colt's foot":  90, "Mallow":  31, "Californian poppy":  68, "Corn poppy":  52, "Moon orchid":  45, "Passion flower":  48, "Grape hyacinth":  78, "Japanese anemone":  66, "Watercress":  72, "Cape flower":  29, "Osteospermum":  77, "Barberton daisy":  20, "Bougainvillea":  27, "Magnolia":  100, "Sunflower":  90, "Daffodil":  12, "Wallflower":  56}

st.title('104 Flowers - Garden of Eden')
st.subheader('Flowers that the models have been trained with and are aware of (104 varities) :')
st.info(list(lanel_dic.keys()))
st.sidebar.info("Created By : Deepthi Sudharsan")
st.sidebar.image("https://storage.googleapis.com/kaggle-datasets-images/514569/948457/98911e4f316dac443b1cbcaf50613840/dataset-card.png?t=2020-02-15-20-31-18", width=None)
st.sidebar.subheader("Github Repo associated with the project : ")
st.sidebar.markdown("[![Github](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJGtP-Pq0P67Ptyv3tB7Zn2ZYPIT-lPGI7AA&usqp=CAU)](https://github.com/DeepthiSudharsan/104-Flower-Classification)")

st.write('Upload a beautiful flower image to predict the type of flower')
 
uploaded_file = st.file_uploader("Choose a flower image ")

def validate_set(img):
    X_valid = []
    image = Image.open(img)     
    image = np.array(image)
    image_data_as_arr = np.asarray(image)
        
    X_valid.append(image_data_as_arr)
    X_valid = np.asarray(X_valid)   
    X_valid = tf.expand_dims(X_valid, axis=-1)
    return X_valid

def prediction(model_name):
    model = tf.keras.models.load_model(model_name)
    y_pred = model.predict(X_val)
    Y_pred_classes = np.argmax(y_pred,axis=1)
    keys = [k for k, v in lanel_dic.items() if v == Y_pred_classes]
    st.subheader("The image is of the flower : ")
    st.success(keys[0])

def download_check(model_url,model_name):
  if(not os.path.isfile(model_name)):
    urllib.request.urlretrieve(model_url,model_name)
  else:
    st.write("Already loaded")
    st.write(os.path.getsize(model_name))
  prediction(model_name)

if(uploaded_file):
  st.write("Uploaded image")
  st.image(uploaded_file)
  X_val = validate_set(uploaded_file)
  # opt = st.selectbox("Select Model(s) ",["Select","MobileNet","ResNet","DenseNet","EfficientNet"])
  opt = st.selectbox("Select Model(s) ",["Select","MobileNet","EfficientNet"])
  if(st.button("Predict")):
    st.write("Model Chosen :")
    st.info(opt)
    if(opt == "Select"):
      st.warning("Select at least one model")
    elif(opt == "EfficientNet"):
      model_url = "https://drive.google.com/uc?export=download&id=1jaYVM8KrNO_QLbdRg25w8-yT-YkWhme5"
      model_name = "eff_model.hdf5"
      download_check(model_url,model_name)
    elif(opt == "MobileNet"):
      model_url = "https://drive.google.com/uc?export=download&id=19DI-ONYkg56f2ZjUAkWf55mD_jsW4e7A"
      model_name = "MobileNet_model.hdf5"
      download_check(model_url,model_name)
    # ResNet and DenseNet work locally, so can be uncommented to work locally
    # elif(opt == "DenseNet"):
    #   model_url = "https://drive.google.com/uc?export=download&id=1VeUeDOCdlGNFI4SEfZ3S7WQlsxGzdICM"
    #   model_name = "DenseNet_model.hdf5"
    #   download_check(model_url,model_name)
    # elif(opt == "ResNet"):
    #   model_url = "https://drive.google.com/uc?export=download&id=1B-0HPl48r8zDZREU5pu3OjJrRkzJYeWP"
    #   model_name = "res_model.h5"
    #   download_check(model_url,model_name)
  st.write("If you would like to remove/delete the models that have been loaded to save up memroy, click the purge button below")
  if(st.button("Purge")):
    if(os.path.isfile("eff_model.hdf5")):
      os.remove("eff_model.hdf5")
    if(os.path.isfile("MobileNet_model.hdf5")):
      os.remove("MobileNet_model.hdf5")
    if(os.path.isfile("DenseNet_model.hdf5")):
      os.remove("DenseNet_model.hdf5")
    if(os.path.isfile("res_model.h5")):
      os.remove("res_model.h5")

else:
  st.warning("No file has been chosen yet")
