import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


# Define the VAE class
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        input_img = tf.keras.Input(shape=(64, 64, 3))
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(input_img)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        z_mean = tf.keras.layers.Dense(self.latent_dim)(x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim)(x)
        return tf.keras.Model(input_img, [z_mean, z_log_var], name="encoder")

    def build_decoder(self):
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(8 * 8 * 64, activation='relu')(latent_inputs)
        x = tf.keras.layers.Reshape((8, 8, 64))(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        decoded = tf.keras.layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)
        return tf.keras.Model(latent_inputs, decoded, name="decoder")

    def reparameterize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed


# Load and preprocess the data
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    r'D:\3-2\DL\new',
    target_size=(64, 64),
    batch_size=32,
    class_mode='input'
)

test_generator = test_datagen.flow_from_directory(
    r'D:\3-2\DL\new',

    target_size=(64, 64),
    batch_size=32,
    class_mode='input'
)

# Create VAE instance
latent_dim = 32
vae = VAE(latent_dim)

# Compile VAE model
vae.compile(optimizer='adam', loss='binary_crossentropy')

# Train VAE model with callbacks
history = vae.fit(train_generator, validation_data=test_generator, epochs=10)

# Evaluate VAE model
loss = vae.evaluate(test_generator)
print("Test Loss:", loss)

# Save the model in TensorFlow SavedModel format
vae.save_weights('vae_weights.h5')

# Visualize original and reconstructed images
num_images = 5
test_images = next(iter(test_generator))[0][:num_images]
reconstructed_images = vae.predict(test_images)

plt.figure(figsize=(10, 4))
for i in range(num_images):
    # Original Images
    plt.subplot(2, num_images, i + 1)
    plt.imshow(test_images[i])
    plt.title("Original")
    plt.axis('off')

    # Reconstructed Images
    plt.subplot(2, num_images, i + 1 + num_images)
    plt.imshow(reconstructed_images[i])
    plt.title("Reconstructed")
    plt.axis('off')

plt.tight_layout()
plt.show()
