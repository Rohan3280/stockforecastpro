from tkinter import *
from tkinter.messagebox import askyesno
from PIL import ImageTk,Image
import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class stock_analysis:
    def __init__(self):
        root=None

    def homepage(self):
        self.root=Tk()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        image1=PhotoImage(file=r"C:\Users\rishi\OneDrive\Desktop\tkinter\back.png")
        Label(self.root,image=image1).place(relheight=1,relwidth=1)
        img=Image.open("juet.jpg")
        img = img.resize((300,300 ))
        img = ImageTk.PhotoImage(img) 
        Label(self.root,image=img).pack(padx=50,pady=50)

       

        def secondpage():
            self.root.destroy()
            self.secondpage1()
        Label(self.root,text="Artificial Intelligence & Application Project",font=("Arial",25,"bold")).pack(padx=20,pady=20)
        Button(self.root,text='SEARCH STOCK', bg='black', fg ='White',font=("Arial",20,"bold"),command=secondpage).pack(pady=20)
        Label(self.root,text="Rahul Sharma (221B291)",font=("Arial",15,"bold")).pack(padx=20,pady=20)
        Label(self.root,text="Rajratan Shivhare (221B292)",font=("Arial",15,"bold")).pack(padx=20,pady=20)
        Label(self.root,text="Raman Soni (221B295)",font=("Arial",15,"bold")).pack(padx=20,pady=20)
        self.root.mainloop()
        
    

    def secondpage1(self):
        
        self.root=Tk()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        image1=PhotoImage(file=r"C:\Users\rishi\OneDrive\Desktop\tkinter\back.png")
        Label(self.root,image=image1).place(relheight=1,relwidth=1)
        img=Image.open("juet.jpg")
        img = img.resize((300,300 ))
        img = ImageTk.PhotoImage(img) 
        Label(self.root,image=img).pack(padx=50,pady=50)


        
        Label(self.root,text='Stock To Analyse',fg='black',bg='white',font=("Arial",20,"bold")).pack(padx=20,pady=20)

        stock=StringVar()
        Entry(self.root,textvariable=stock,width=20,font=("Arial",25,"bold")).pack(padx=20,pady=20)

        def firstpage():
            self.root.destroy()
            self.homepage()
        

        Button(self.root,text='Go  TO  HOMEPAGE', bg= 'black', fg ='white',command=firstpage,font=("Arial",20,"bold")).pack(padx=20,pady=20)

        def reliance():
            quandl.ApiConfig.api_key='oaUWCMQ1xq4xHmvLtagd'

            so=stock.get()
            
            df=quandl.get(so)
            #print(df.head(15))

            plt.figure(figsize=(16,8))
            plt.plot(df['Close'], label='Closing price')
            plt.show()



        

                    
            df['Open - Close']=df['Open']-df['Close']
            df['High - Low'] = df['High']-df['Low']
            df=df.dropna()

            X=df[['Open - Close','High - Low']]
            X.head()
            #print(X)

            Y= np.where(df['Close'].shift(-1)>df['Close'],1,-1)
            #print(Y)

            from sklearn.model_selection import train_test_split
            X_train ,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=44)





            from sklearn.neighbors import KNeighborsClassifier
            from sklearn import neighbors
            from sklearn.model_selection import GridSearchCV
            from sklearn.metrics import accuracy_score

            params={'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
            knn =neighbors.KNeighborsClassifier()
            model =GridSearchCV(knn,params,cv=5)

            model.fit(X_train,Y_train)

            accuracy_train = accuracy_score(Y_train,model.predict(X_train))
            accuracy_test = accuracy_score(Y_test,model.predict(X_test))

            accuracy_train=accuracy_train*100
            print('Train data Accuracy is %.3f' %accuracy_train)
            #print('Test data Accuracy is %.3f' %accuracy_test)

            predctions_classification =model.predict(X_test)
            actual_predicted_data =pd.DataFrame({'Actual Class':Y_test, 'Predicted class ':predctions_classification})
            #print(actual_predicted_data.head(1))
            rt=predctions_classification[-1]
            if(rt==1):
                #print("BUY")
                Label(self.root,text='BUY',font=("Arial",20,"bold"),fg="green",bg="black").pack(padx=10,pady=10)
            else:
                #print("SELL") 
                Label(self.root,text='SELL ',font=("Arial",20,"bold"),fg="red",bg="black").pack(padx=10,pady=10)   

            Y=df['Close']
            #print(Y)    



            from sklearn.neighbors import KNeighborsRegressor
            from sklearn import neighbors
            X_train_reg ,X_test_reg,Y_train_reg,Y_test_reg = train_test_split(X,Y,test_size=0.25,random_state=44)


            params={'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
            knn_reg =neighbors.KNeighborsRegressor()
            model_reg =GridSearchCV(knn_reg,params,cv=5)
            model_reg.fit(X_train_reg,Y_train_reg)
            predctions =model_reg.predict(X_test_reg)

            rrm=predctions
            #print(predctions[-1])
            Label(self.root,text=predctions[-1],font=("Arial",20,"bold"),fg="white",bg="black").pack()



        Button(self.root,text='Generate Chart', bg= 'black',fg= 'white',command=reliance,font=("Arial",20,"bold")).pack(padx=20,pady=20)
        
            


        self.root.mainloop()

        
obj=stock_analysis().homepage()


