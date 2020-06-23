import tkinter as tk
from tkinter import * 
from tkinter import Message ,Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font


root = tk.Tk()
root.geometry("900x300")
root.minsize(300,200)
root.title("Employee Attendance")


root.configure(background='white')



 
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def ImageCapture():        
    Id=txtVar.get()
    print(Id)
    name=txt2Var.get()
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\\"+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('EmployeeDetails\\EmployeeDetails.csv','a+') as csvFile: # a+ is appending after line
            writer = csv.writer(csvFile) #create writer object
            writer.writerow(row) #row wise write
        csvFile.close()
        messageVar.set(str(res))
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            messageVar.set(str(res))
        if(name.isalpha()):
            res = "Enter Numeric Id"
            messageVar.set(str(res))
    
def ModelTrain():
	
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\\Trainner.yml")
    res = "Image Trained"
    messageVar.set(str(res))

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("EmployeeDetails\\EmployeeDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("UnrecognizedImages"))+1
                cv2.imwrite("UnrecognizedImages\\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    res=attendance
    a= str(res)
    messageVar.set(a)





#variable set
txtVar = StringVar()
txt2Var = StringVar()
messageVar = StringVar()
message2Var = StringVar()

lbl = tk.Label(root,text="Enter ID",width=15  ,height=1  ,fg="white"  ,bg="gray" ,font=('times', 15, ' bold ') ).grid(row=0, column=0,pady=4)
txt = tk.Entry(root,textvariable=txtVar, width=24 ,bg="gray" ,fg="white",font=('times', 15, ' bold ')).grid(row=0, column=1,padx=4)

lbl2 = tk.Label(root, text="Enter Name",width=15  ,height=1  ,fg="white"  ,bg="gray" ,font=('times', 15, ' bold ') ).grid(row=1, column=0)
txt2 = tk.Entry(root,textvariable=txt2Var, width=24  ,bg="gray" ,fg="white",font=('times', 15, ' bold ')).grid(row=1, column=1,padx=4)

lbl3 = tk.Label(root, text="Notification",width=15  ,height=1 ,fg="white"  ,bg="gray" ,font=('times', 15, ' bold ') ).grid(row=2, column=0,pady=4)
message = tk.Label(root,textvariable=messageVar, width=35  ,bg="gray" ,fg="white",font=('times', 12, ' bold ')).grid(row=2, column=1,padx=4)



trainImg = tk.Button(root, text="Identify Person",command=TrackImages,width=20  ,height=1  ,fg="white"  ,bg="gray" ,font=('times', 15, ' bold ') ).grid(row=4, column=1,pady=4,padx=4)

def buttonClicked(btn):
    if(btn == "Image Capture"):
        ImageCapture()
    elif(btn == "Model Train"):
        ModelTrain()    

btnMenu = Menubutton(root, text='New Employee Enroll',width=20  ,height=1  ,fg="white"  ,bg="gray" ,font=('times', 15, ' bold ') )
contentMenu = Menu(btnMenu)
btnMenu.config(menu=contentMenu)
btnMenu.grid(row=4, column=0,pady=4,padx=4)

btnList = ['Image Capture', 'Model Train']
for btn in btnList:
    contentMenu.add_command(label=btn, command = lambda btn=btn: buttonClicked(btn))
 
root.mainloop()