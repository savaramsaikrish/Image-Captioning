import os
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


class preprocess_data:
    # Write method to visualize images
    def visualize_images(self, dir_path, nimages):
        fig, axs = plt.subplots(3, 5, figsize=(10, 10))  # 2 rows, 5 columns for 10 images
        dpath = dir_path
        count = 0
        for i in os.listdir(dpath):
            # Get the list of images in a given class
            train_class = os.listdir(os.path.join(dpath, i))
            # Plot the images, but ensure we don't go out of bounds
            for j in range(min(nimages, len(train_class))):  # Limit to the length of train_class
                img_path = os.path.join(dpath, i, train_class[j])
                img = cv2.imread(img_path)
                axs[count][j].title.set_text(i)
                axs[count][j].imshow(img)
            count += 1
        fig.tight_layout()
        plt.show(block=True)

    # Write method to preprocess the data
    def preprocess(self, dir_path):
        dpath = dir_path
        train = []
        labels = []
        for i in os.listdir(dpath):
            # Get the list of images in a given class
            train_class = os.listdir(os.path.join(dpath, i))
            for j in train_class:
                img = os.path.join(dpath, i, j)
                train.append(img)
                labels.append(i)
        print("Number of images: {}\n".format(len(train)))
        print("Number of image labels: {}\n".format(len(labels)))
        leaf_df = pd.DataFrame({'Image': train, 'Labels': labels})
        print(leaf_df)
        return leaf_df, train, labels

    # Skill4 ==> Image data generator
    def generate_train_test_split(self, retina_df, train, labels):
        train_df, test_df = train_test_split(retina_df, test_size=0.2)

        train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, validation_split=0.2)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(128, 128),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=12,
            subset='training'
        )

        validate_generator = train_datagen.flow_from_dataframe(
            train_df,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(128, 128),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=12,
            subset='validation'
        )

        test_generator = test_datagen.flow_from_dataframe(
            test_df,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(128, 128),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=12
        )

        print(f"Train image shape: {train_df.shape}")
        print(f"Test image shape: {test_df.shape}")

        return train_generator, test_generator, validate_generator
