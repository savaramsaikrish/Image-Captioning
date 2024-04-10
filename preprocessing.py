import os
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import scipy



class preprocess_data:
    #write method to visualize images
    def visualize_images(self,dir_path,nimages):
        fig, axs = plt.subplots(1, 5, figsize = (10, 10))
        dpath=dir_path
        count=0
        for i in os.listdir(dpath):
            #get the list of images in a given class
            train_class=os.listdir(os.path.join(dpath,i))
            #plot the images
            for j in range(min(nimages, len(train_class))):
                img=os.path.join(dpath,i,train_class[j])
                img=cv2.imread(img)
                axs[count][j].title.set_text(i)
                axs[count][j].imshow(img)
            count+=1
        fig.tight_layout()
        plt.show(block=True)

    #write method to preprocess the data
    def preprocess(self, dir_path):
        dpath=dir_path
        #count the number of images in the dataset
        train=[]
        labels=[]
        for i in os.listdir(dpath):
            #get the list of images in a given class
            train_class=os.listdir(os.path.join(dpath,i))
            for j in train_class:
                img=os.path.join(dpath,i,j)
                train.append(img)
                labels.append(i)
        print("number of images:{}\n".format(len(train)))
        print("number of image labels:{}\n".format(len(labels)))
        retina_df=pd.DataFrame({'Image':train, 'Labels':labels})
        print(retina_df)
        return retina_df,train,labels


    # for generate the more images or split the data into strain,test,validation

    def generate_train_test_split(self,retina_df, train, label):

        train, test=train_test_split(retina_df,test_size=0.2)

        train_datagen = ImageDataGenerator(rescale= 1. /255, shear_range=0.2, validation_split=0.15)

        test_datagen = ImageDataGenerator(rescale= 1./255)

        train_generator= train_datagen.flow_from_dataframe(

            train,
            directory='./',
            y_col="Labels",
            x_col="Image",
            target_size=(28,28),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32,

            subset='training'
        )

        validation_generator = train_datagen.flow_from_dataframe(

            train,
            directory='./',
            y_col="Labels",
            x_col="Image",
            target_size=(28,28),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32,

            subset='validation'
        )

        test_generator = train_datagen.flow_from_dataframe(

            test,
            directory='./',
            y_col="Labels",
            x_col="Image",
            target_size=(28,28),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32,

        )

        print(f"Train images shape: {train.shape}")
        print(f"testing image shape : {test.shape}")
        return train_generator,test_generator,validation_generator