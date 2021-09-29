### 104 Flowers - Garden of Eden (Image Classification)

This project has been made as a part of TMLC's (The Machine Learning Company's) Deep Learning Program

### AIM

The goal is to classify the variety of flower uploaded by the user. Pretrained models Resnet, DenseNet, EfficientNet and MobileNet have been used for classification.

[NOTE : BECAUSE PRE-TRAINED MODELS RESNET AND DENSENET THAT HAVE BEEN USED IN THE PROJECT ARE TOO LARGE TO BE LOADED ON STREAMLIT SHARE, THE CODE FOR THE TWO MODELS HAVE BEEN COMMENTED OUT IN THE .PY FILE. WHILE RUNNING LOCALLY, IF YOU WANT TO TRY THOSE MODELS, BE SURE TO UNCOMMENT ALL THE COMMENTED CODE PARTS AND SEE THE MAGIC HAPPEN]

### DATA

Data with which the models were trained can be found here :

https://drive.google.com/file/d/1QDN2eQ1_mnFEaxMLS5dYT9WJXz5p3n_v/view?usp=sharing

### STREAMLIT GUI

This app has been deployed on streamlit. To view the app check the link below

https://share.streamlit.io/deepthisudharsan/104-flower-classification/main/flower_app.py

### Pre-requisites to run Streamlit app locally :

Make sure to install streamlit if haven't already, to install streamlit use the following command :

```
pip install streamlit
```
All the package requirements along with the versions have been mentioned in the requirements.txt file. Running the code is as simple as going to your Anaconda Prompt, navigating to the directly with your streamlit py files, and running the command 
```
$ streamlit run flower_app.py
```
### How to run?

* Clone the repository
* Setup Virtual environment
```
$ python3 -m venv env
```
* Activate the virtual environment and go to the streamlit folder
```
$ source env/bin/activate
```
* Install dependencies using
```
$ pip install -r requirements.txt
```
* Run Streamlit
```
$ streamlit run flower_app.py
```

### SNIPPETS FROM THE APP

![image](https://user-images.githubusercontent.com/59824729/135339596-9f518d5d-e409-419a-bac6-817c36c9bab2.png)
![image](https://user-images.githubusercontent.com/59824729/135339632-a98b4b52-389c-487a-a473-0981b31f1d69.png)
![image](https://user-images.githubusercontent.com/59824729/135339688-1b600b41-2a53-4c39-8d7c-e4f5f6e5fcde.png)
