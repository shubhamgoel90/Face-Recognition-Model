import cv2
import numpy as np
import os

########## KNN CODE ############
def distance(v1, v2):
	# Eucledian 
	return np.sum((v2 - v1)**2)**0.5  


def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]
################################

cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

dataset_path ="./FaceData/"
labels = []
class_id = 0           # Labels for the given file
names = {}             # Mapping between id - name
face_data = []


# Data Preparation
for fx in os.listdir(dataset_path):
	if fx.endswith(".npy"):
		names[class_id] = fx[:-4]
		print("Loading file " + fx)
		data_item = np.load(dataset_path+fx)
		face_data = face_data
		face_data.append(data_item)
		#Create Labels for the class
		target = class_id*np.ones((data_item.shape[0],))
		class_id +=1 
		labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels= np.concatenate(labels,axis=0).reshape((-1,1))




print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

#Testing
while True:
	#Prediction _____!!!

	ret, frame = cam.read()
	if ret == False:
		continue


	faces = face_cascade.detectMultiScale(frame,1.3,5)   # faces is the list whcih has tuples. print(faces) will show the cordinates of face at every second. Basically where are face it is. eg [(255,283,32)]
	if len(faces)==0:
		continue
	
	# Pick the largest face (beacuse it is biggest according to the area ( f[2]*f[3] ))
	for face in faces:
		x,y,w,h = face
		#cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255, 255),2)
		
		# Extract (Crop of the required image) : Region of intrest
		face_section = frame[y-10:y+h+10, x-10: x+w+10]
		face_section = cv2.resize(face_section, (100,100))
		#cv2.rectangle(frame,(x,y),(x+w,y+h),(0 ,255, 255,2))
		
		#predicted Label
		out = knn(trainset, face_section.flatten())

		# Display on the screen and rectangle around it.
		pred_name = names[int(out)]
		cv2.putText(frame, pred_name,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1 ,(255, 0 ,0), 2,cv2.LINE_AA)
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255,255), 2)


	cv2.imshow("Faces", frame)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()


	





