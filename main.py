import streamlit as st
import pickle
import numpy as np
import logging as l
from PIL import Image

model = pickle.load(open('Random_forest_model.pkl', 'rb'))

l.basicConfig(filename='log.txt',filemode='a',format='%(asctime)s-%(levelname)s-%(message)s',datefmt='%Y-%m-%d %H:%M:%S')

try:
    st.title('Mushroom Classifier')
    best_features={'cap-shape':['bell','conical','convex','flat','knobbed','sunken'], 'bruises':['bruises','no'], 'odor':['almond','anise','creosote','fishy','foul','musty','none','pungent','spicy'], 'gill-spacing':['close','crowded','distant'], 'gill-size':['broad','narrow'], 'gill-color':['black','brown','buff','chocolate','gray','green','orange','pink','purple','red','white','yellow'], 'stalk-root':['bulbous','club','cup','equal','rhizomorphs','rooted','missing'], 'stalk-surface-above-ring':['fibrous','scaly','silky','smooth'], 'stalk-surface-below-ring':['fibrous','scaly','silky','smooth'], 'stalk-color-above-ring':['brown','buff','cinnamon','gray','orange','pink','red','white','yellow'], 'stalk-color-below-ring':['brown','buff','cinnamon','gray','orange','pink','red','white','yellow'], 'ring-type':['cobwebby','evanescent','flaring','large','none','pendant','sheathing','zone'], 'spore-print-color':['black','brown','buff','chocolate','green','orange','purple','white','yellow'], 'population':['abundant','clustered','numerous','scattered','several','solitary'], 'habitat':['grasses','leaves','meadows','paths','urban','waste','woods']}
    st.write('Classifier')
    values={}
    for i in best_features:
        values[i]=st.selectbox(i,best_features[i])
except:
    l.warning('cannot create the form' )

if(st.button('Predict')):
    #print(values)


    ## matching and replacing with its relevant characters as it was in data set

    try:
        dict={'bell': 'b', 'conical': 'c', 'convex': 'x', 'flat': 'f', ' knobbed': 'k', 'sunken': 's', 'fibrous': 'f', 'grooves': 'g', 'scaly': 'y', 'smooth': 's', 'brown': 'n', 'buff': 'b', 'cinnamon': 'c', 'gray': 'g', 'green': 'r', 'pink': 'p', 'purple': 'u', 'red': 'e', 'white': 'w', 'yellow': 'y', 'bruises': 't', 'no': 'f', 'almond': 'a', 'anise': 'l', 'creosote': 'c', 'fishy': 'y', 'foul': 'f', 'musty': 'm', 'none': 'n', 'pungent': 'p', 'spicy': 's', 'attached': 'a', 'descending': 'd', 'free': 'f', 'notched': 'n', 'close': 'c', 'crowded': 'w', 'distant': 'd', 'broad': 'b', 'narrow': 'n', 'black': 'k', 'chocolate': 'h', ' green': 'r', 'orange': 'o', 'enlarging': 'e', 'tapering': 't', 'bulbous': 'b', 'club': 'c', 'cup': 'u', 'equal': 'e', 'rhizomorphs': 'z', 'rooted': 'r', 'missing': '?', 'silky': 'k', 'partial': 'p', 'universal': 'u', 'one': 'o', 'two': 't', 'cobwebby': 'c', 'evanescent': 'e', 'flaring': 'f', 'large': 'l', 'pendant': 'p', 'sheathing': 's', 'zone': 'z', 'abundant': 'a', 'clustered': 'c', 'numerous': 'n', 'scattered': 's', 'several': 'v', 'solitary': 'y', 'grasses': 'g', 'leaves': 'l', 'meadows': 'm', 'paths': 'p', 'urban': 'u', 'waste': 'w', 'woods': 'd'}
        for i in values:
            if(values[i] in dict):
                values[i]=dict[values[i]]
        print(values)
    except:
        l.critical('Error occured during matching and replacing with its relevant characters as it was in data set')

    ## encoding the converted data
    try:
        encode={'class': ['e', 'p'],'cap-shape': ['b', 'c', 'f', 'k', 's', 'x'],'cap-surface': ['f', 'g', 's', 'y'],'cap-color': ['b', 'c', 'e', 'g', 'n', 'p', 'r', 'u', 'w', 'y'],'bruises': ['f', 't'],'odor': ['a', 'c', 'f', 'l', 'm', 'n', 'p', 's', 'y'],'gill-attachment': ['a', 'f'],'gill-spacing': ['c', 'w'],'gill-size': ['b', 'n'],'gill-color': ['b', 'e', 'g', 'h', 'k', 'n', 'o', 'p', 'r', 'u', 'w', 'y'],'stalk-shape': ['e', 't'],'stalk-root': ['?', 'b', 'c', 'e', 'r'],'stalk-surface-above-ring': ['f', 'k', 's', 'y'],'stalk-surface-below-ring': ['f', 'k', 's', 'y'],'stalk-color-above-ring': ['b', 'c', 'e', 'g', 'n', 'o', 'p', 'w', 'y'],'stalk-color-below-ring': ['b', 'c', 'e', 'g', 'n', 'o', 'p', 'w', 'y'],'veil-type': ['p'],'veil-color': ['n', 'o', 'w', 'y'],'ring-number': ['n', 'o', 't'],'ring-type': ['e', 'f', 'l', 'n', 'p'],'spore-print-color': ['b', 'h', 'k', 'n', 'o', 'r', 'u', 'w', 'y'],'population': ['a', 'c', 'n', 's', 'v', 'y'],'habitat': ['d', 'g', 'l', 'm', 'p', 'u', 'w']}
        arr=[]
        #c=1
        for i in values:
            if i in encode:
                x=values[i]
                #print(x)
                if(x in encode[i]):
                    #print(c,encode[i].index(x),x,sep=":")
                    arr.append(encode[i].index(x))
                elif(not(x in encode[i])):
                    #print(c,len(encode[i]),sep=':')
                    arr.append(len(encode[i]))
                #c+=1
    except:
        l.critical('Error occured during encoding the converted data')

    ## Converting the encoded data into Numpy array

    arr = np.array([arr])

    ## predict the value

    try:
        pred = model.predict(arr)
    except:
        l.critical('Error occured during prediction')

    ##print image based on the prediction
    print(*arr,sep=',')

    if(pred[0]==0):
        img=Image.open('img/edible-modified.png')
        st.image(img,caption='Edible')
    elif(pred[0]==1):
        img = Image.open('img/poison-modified.png')
        st.image(img, caption='Poisonous')