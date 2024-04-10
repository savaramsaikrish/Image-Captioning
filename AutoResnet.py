from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
# Set up image data generators for train and test datasets
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
#image size is changed to 32 as resnet receives 32 ,32 image size
train_generator = train_datagen.flow_from_directory(
    r'D:\3-2\DL\new',
    target_size=(32, 32),
    batch_size=32,
    class_mode='input'  # This assumes you want the input image as the output for the autoencoder
)

test_generator = test_datagen.flow_from_directory(
    r'D:\3-2\DL\new',
    target_size=(32, 32),
    batch_size=32,
    class_mode='input'
)

# Define autoencoder architecture
input_img = Input(shape=(32, 32, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Create autoencoder model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder using the train data generator
autoencoder.fit(train_generator, epochs=2,validation_data=test_generator)

autoencoder.save('autoencoder_with_resnet.h5')
# Evaluate the model on the test dataset
loss = autoencoder.evaluate(train_generator)
print("Test loss:", loss)

model = load_model('autoencoder_with_resnet.h5')
nimages=10

# Load a batch of images from the test generator
#test_images, _ = next(test_generator)

# Predict images using the autoencoder
test_images = autoencoder.predict(test_generator)
total_samples = len(train_generator)
all_images = []

# Iterate over the generator to extract all images
for i in range(total_samples):
    # Get the next batch of data from the generator
    images, _ = next(train_generator)
    # Append the batch of images to the list
    all_images.extend(images)

# Convert the list of images to a numpy array
train_images = np.array(all_images)
x_train_encoded=autoencoder.predict_generator(train_images)
print("extracted features shape", x_train_encoded.shape)
 #Define ResNet model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32,32,3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
resnet_model = Model(inputs=base_model.input, outputs=predictions)
#Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

resnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using the features extracted by autoencoder
resnet_model.fit(x_train_encoded,
          train_generator.classes,
          batch_size=32,
          epochs=10)

# Predictions
predictions = resnet_model.predict(x_train_encoded)


# Evaluate model
score = resnet_model.evaluate(test_images)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
num_images=10
# Display original images, compressed images, and classifications
fig, axs = plt.subplots(2, num_images, figsize=(10, 10))
for i in range(num_images):
            #   # Original image
            axs[0, i].imshow(train_images[i])
            axs[0, i].set_title('Original')
            axs[0, i].axis('off')
            # compressed  image
            axs[1, i].imshow(x_train_encoded[i])
            axs[1, i].set_title('compressed')
            axs[1, i].axis('off')
            #print(predictions[i])
            #print(train_generator.classes[i])
plt.show(block=True)