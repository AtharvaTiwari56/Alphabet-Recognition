import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
from PIL import Image
import PIL.ImageOps
import cv2
import numpy as np

x = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']

classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses=len(classes)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 7500, test_size = 2500)

x_train = x_train/255.0
x_test = x_test/255.0

logreg = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_train, y_train)
y_pred = logreg.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

cam = cv2.VideoCapture(0)

while(True):
    try:
        ret, current_frame = cam.read()

        grayscale = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        height, width = grayscale.shape
        upper_left = (int(width/2 - 56), int(height/2-56))
        bottom_right = (int(width/2+56), int(height/2+56))
        cv2.rectangle(grayscale, upper_left, bottom_right, (0, 255, 0), 2)

        roi = grayscale[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
        
        pil_img = Image.fromarray(roi)
        
        image_bw = pil_img.convert('L')
        image_bw_resize = image_bw.resize((28, 28), Image.ANTIALIAS)
        
        img_bw_rsz_invert = PIL.ImageOps.invert(image_bw_resize)

        pixel_filter = 20
        min_pixel = np.percentile(img_bw_rsz_invert, pixel_filter)

        im_bw_rs_inv_scaled = np.clip(img_bw_rsz_invert-min_pixel, 0, 255)
        
        max_pixel = np.max(img_bw_rsz_invert)

        

        im_bw_rs_inv_scaled = np.asarray(im_bw_rs_inv_scaled)
        test_sample = np.array(im_bw_rs_inv_scaled).reshape(1, 784)
        test_pred = logreg.predict(test_sample)
        print('test_pred',test_pred)

        cv2.imshow('number', grayscale)
        
        if cv2.waitKey(1) == ord('q'):
            break
    except:
        pass

cam.release()
cv2.destroyAllWindows()