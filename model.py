import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#tf.keras.datasets: Datasets which can be loaded and used for training and testing ML models
#mnist: Dataset of 70,000 handwritten digits (0-9) --> Use to train classification DL models
mnist = tf.keras.datasets.mnist

#Split into training and testing data
#X will contain the digits and Y will contain the classification
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalize and Scaling the Data --> Scailing the pixel value to be between 0-1
#Normalization of the data would make training more efficient --> divide input values by Euclidean norm
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)


#Loading and Training models
#model = tf.keras.models.Sequential()

#Add flat layer
#model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

#Add dense layer
#model.add(tf.keras.layers.Dense(128,activation = 'relu'))
#model.add(tf.keras.layers.Dense(10,activation = 'softmax'))



#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics = ['accuracy'])
#model.fit(x_train,y_train, epochs= 3)
#model.save('handwritten_model.keras')


model = tf.keras.models.load_model('handwritten_model.keras')


image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1
