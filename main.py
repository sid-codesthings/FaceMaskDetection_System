import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
st.title("FACE MASK DETECTION SYSTEM")
st.sidebar.image("https://cdn.imgbin.com/1/6/15/imgbin-facial-recognition-system-computer-icons-face-detection-iris-recognition-scanner-y78VcgcD5aV3BTReGGiizifnQ.jpg")
choice=st.sidebar.selectbox("Menu",("HOME","URL","CAMERA"))
st.header(choice) # st.header is used for printing the choice
if(choice=="HOME"):
    st.image("https://5.imimg.com/data5/PI/FD/NK/SELLER-5866466/images-500x500.jpg")
elif(choice=="URL"):
    url=st.text_input("Enter your URL")
    btn=st.button("Start Detection")
    window=st.empty() # for 
    if btn:
        i=1
        btn2 = st.button("Stop Detection")
        if btn2:
            st.experimental_rerun() # for rereun.
        facemodel=cv2.CascadeClassifier("face.xml")
        maskmodel=load_model("mask.h5")
        vid=cv2.VideoCapture(url)
        while(vid.isOpened()):
            flag,frame=vid.read()
            if flag:
                faces=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in faces: # w or h is same.
                    face_img=frame[y:y+w,x:x+l] # cropping
                    face_img=cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
                    face_img=np.asarray(face_img,dtype=np.float32).reshape(1,224,224,3)
                    face_img=(face_img/127.5)-1
                    p=maskmodel.predict(face_img)[0][0]
                    if(p>0.9):
                        path="nomask/"+str(i)+".jpg" # saving without mask faces in nomask folder of our environment as 1.jpg , 2.jpg etc
                        cv2.imwrite(path,frame[y:y+w,x:x+l]) # saving the cropped unmasked faces.
                        i=i+1
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)
                window.image(frame,channels='BGR')
elif(choice=="CAMERA"):
    cam = st.selectbox("Choose Camera",("None","Primary","Secondary"))
    btn=st.button("Start Detection")
    window=st.empty() # for 
    if btn:
        i=1
        btn2 = st.button("Stop Detection")
        if btn2:
            st.experimental_rerun() # for rereun.
        facemodel=cv2.CascadeClassifier("face.xml")
        maskmodel=load_model("mask.h5")
        if cam=="Primary":
            cam=0 # for primray cam i.e. front cam
        else:
            cam=1
        vid=cv2.VideoCapture(cam)
        while(vid.isOpened()):
            flag,frame=vid.read()
            if flag:
                faces=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in faces: # w or h is same.
                    face_img=frame[y:y+w,x:x+l] # cropping
                    face_img=cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
                    face_img=np.asarray(face_img,dtype=np.float32).reshape(1,224,224,3)
                    face_img=(face_img/127.5)-1
                    p=maskmodel.predict(face_img)[0][0]
                    if(p>0.9):
                        path="nomask/"+str(i)+".jpg" # saving without mask faces in nomask folder of our environment as 1.jpg , 2.jpg etc
                        cv2.imwrite(path,frame[y:y+w,x:x+l]) # saving the cropped unmasked faces.
                        i=i+1
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)
                window.image(frame,channels='BGR')
 
    
