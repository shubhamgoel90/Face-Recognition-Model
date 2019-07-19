import cv2
import numpy as np


#Read a Video Stream and Display It

#Camera Object
cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
face_data= []
cnt =0
user_name = input("Enter your name")


while True:
	ret,frame = cam.read()
	if ret==False:
		print("Something Went Wrong!")
		continue

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

	faces = face_cascade.detectMultiScale(frame,1.3,5)   # faces is the list whcih has tuples. print(faces) will show the cordinates of face at every second. Basically where are face it is. eg [(255,283,32)]
	faces = sorted(faces , key = lambda f : f[2]*f[3])   # Sorting b/c we want to save the biggest face in the video for accuracy as if two faces come in the video only one will be stored.
	
	if (len(faces)==0):
		continue
	
	# Pick the largest face (beacuse it is biggest according to the area ( f[2]*f[3] ))
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255, 255),2)
		
		# Extract (Crop of the required image) : Region of intrest
		face_section = frame[y-10:y+h+10, x-10: x+w+10]
		face_section = cv2.resize(face_section, (100,100))
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0 ,255, 255,2))
		
		if cnt%10 == 0:
			print("Taking pictures",int(cnt/10))
			face_data.append(face_section)
		cnt+=1
	


	
	cv2.imshow("Video", frame)
	cv2.imshow("Video gray", face_section)

# save the data in the numpy file
print("Total faces",len(face_data))
face_data = np.array(face_data)
face_data = face_data.reshape([face_data.shape[0],-1])
print(face_data.shape)
np.save("FaceData/" + user_name+ ".npy",face_data)  # it save the image into array form so that we can use it later
print(face_data.shape)


cam.release()
cv2.destroyAllWindows()
