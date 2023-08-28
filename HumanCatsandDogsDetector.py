import numpy as np
import cv2
from matplotlib import pyplot as plt

dog_cascade=cv2.CascadeClassifier('dogdetector.xml')
face_cascade=cv2.CascadeClassifier('facedetector.xml')
cat_cascade=cv2.CascadeClassifier('catdetector.xml')

img=cv2.imread('family.bmp')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
font=cv2.FONT_HERSHEY_SIMPLEX
dogs=dog_cascade.detectMultiScale(gray,1.345,5,75)
faces=face_cascade.detectMultiScale(gray,1.3,5)
cats=cat_cascade.detectMultiScale(gray,1.3,2,75)

for(x,y,w,h) in dogs:
	img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	cv2.putText(img,'Dog',(x,y),font,0.9,(0,255,0),2)

for(z,v,b,n) in faces:
	img=cv2.rectangle(img,(z,v),(z+b,v+n),(0,0,255),2)
	cv2.putText(img,'Human',(z,v),font,0.9,(0,0,255),2)

for(q,w,e,r) in cats:
	img=cv2.rectangle(img,(q,w),(q+e,w+r),(255,0,0),2)
	cv2.putText(img,'Cat',(q,w),font,0.9,(255,0,0),2)
	
p,l,m=cv2.split(img)
img=cv2.merge([m,l,p])

plt.imshow(img)
plt.show()


# video_capture = cv2.VideoCapture(0)

# def detect_bounding_box(vid):
#     gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
#     dogs=dog_cascade.detectMultiScale(gray,1.345,5,75)
#     faces=face_cascade.detectMultiScale(gray,1.3,5)
#     cats=cat_cascade.detectMultiScale(gray,1.3,2,75)    
# for(x,y,w,h) in dogs:
# 	img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# 	cv2.putText(img,'Dog',(x,y),font,0.9,(0,255,0),2)

# for(z,v,b,n) in faces:
# 	img=cv2.rectangle(img,(z,v),(z+b,v+n),(0,0,255),2)
# 	cv2.putText(img,'Human',(z,v),font,0.9,(0,0,255),2)

# for(q,w,e,r) in cats:
# 	img=cv2.rectangle(img,(q,w),(q+e,w+r),(255,0,0),2)
# 	cv2.putText(img,'Cat',(q,w),font,0.9,(255,0,0),2)

# p,l,m=cv2.split(img)
# img=cv2.merge([m,l,p])

# while True:

#     result, video_frame = video_capture.read()  # read frames from the video
#     if result is False:
#         break  # terminate the loop if the frame is not read successfully

#     faces = detect_bounding_box(
#         video_frame
#     )  # apply the function we created to the video frame

#     cv2.imshow(
#         "My Face Detection Project", video_frame
#     )  # display the processed frame in a window named "My Face Detection Project"

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# video_capture.release()

cv2.waitKey(0)
cv2.destroyAllWindows()

