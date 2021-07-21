from flask import Flask, render_template, request
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import PIL
from PIL import UnidentifiedImageError
app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 2


@app.route('/')
def index():
    return render_template('index.html')





@app.route('/', methods=['GET', 'POST'])
def after():
    result = False
    #if request.files['file1'] is None:
    #    abort(400)
    #    return render_template('index.html')
    img1 = request.files['file1'] 
    img1.save('static/file.jpg')
    
    #if request.files['file2'] is None:
    #    abort(400)
    #    return render_template('index.html')
    img2 = request.files['file2']
    img2.save('static/file2.jpg')

    ####################################
    img1 = cv.imread('static/file.jpg',0)          # queryImage
    img2 = cv.imread('static/file2.jpg',0) # trainImage
    
    try:
      image1 = PIL.Image.open('static/file.jpg')
      image2 = PIL.Image.open('static/file2.jpg')
    except UnidentifiedImageError:
      return render_template('index.html')
    
    width1, height1 = image1.size
    width2, height2 = image2.size
    if width1 == width2 and height1 ==height2:
        difference = cv.subtract(img1, img2)    
        result = not np.any(difference)

    method = 'SIFT'  # 'SIFT'
    #method = 'ORB'
    lowe_ratio = 0.725
    
    if method   == 'ORB':
        finder = cv.ORB_create()
    elif method == 'SIFT':
        finder = cv.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
    kp1, des1 = finder.detectAndCompute(img1,None)
    kp2, des2 = finder.detectAndCompute(img2,None)

# BFMatcher with default params
    #bf = cv.BFMatcher(cv2.NORM_L2, crossCheck=True)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    #matches = bf.Match(des1,des2)

# Apply ratio test
    good = []
    #print(img1)
    #print(img2)
    #an_array = np.array(img1)
    #another_array = np.array(img2)
    #comparison = an_array.all() == another_array.all()
    
    #print("comparison" +str(comparison))
    #equal_arrays = comparison.all()
    for m,n in matches:
        if m.distance < lowe_ratio*n.distance:
            good.append([m])
#dist = 1 - len(good) / (max(len(img1.description), len(img2.description)))
    if len(good)>7:
        n=11
    else:
        n=5
    msg1 = 'using %s with lowe_ratio %.2f' % (method, lowe_ratio)
    score = (len(good)/n*100)
    if result is True:
        score = 100
    elif score >=100:
        score = 99
    msg2 = '%d' % score + '% similarity score'

    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    font = cv.FONT_HERSHEY_SIMPLEX
#cv.putText(img3,msg1,(20, 150), font, 0.3,(255,255,255),1,cv.LINE_AA)
    #cv.putText(img3,msg2,(5, 190), font, 0.5,(255,255,255),1, cv.FILLED)

    fname = 'static/after.jpg' 
    cv.imwrite(fname, img3)

    return render_template('index.html',data=msg2)

if __name__ == "__main__":
    app.run(debug=True)


