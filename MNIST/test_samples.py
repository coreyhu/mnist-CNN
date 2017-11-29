from keras.models import load_model
import scipy
import numpy as np
from keras.utils import np_utils


model = load_model('mnist_cnn.h5')

digits_list = ['samples/1.jpg', 'samples/2.jpg', 'samples/3.jpg', 'samples/4.jpg', 'samples/7.jpg', 'samples/9.jpg']
digits_labels = [1, 2, 3, 4, 7, 9]

digits_ohe = np_utils.to_categorical(digits_labels, 10)

digits = []
for digit_location in digits_list:
    img = scipy.ndimage.imread(digit_location, flatten=True)
    img = img.reshape(28, 28, 1)
    digits.append(img)

digits = np.asarray(digits)


y_pred = model.predict(digits)

score = model.evaluate(digits, digits_ohe, verbose=1)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])
