# Face-Recognition-Model
This model is created in two parts. In first part ie in this "face_part1_detection" file we are doing a live stream and entering the name of the person in the live stream then using "haarcascade_frontalface_alt" file we are detecting the face in the live stream and then cropping only the one bigger face ie which is the main face in the live stream and saving that image in the form of pixels and converting it into .npy extension and saving it into FaceData folder.

Now, in the second part ie in "face_part2_detection" we are going to the FaceData folder and picking all the files and then taking thier pixel values and then again doing live stream and with help of KNN Classifier we are detecting the faces with name of that person displayed over the face with a rectangular box around the face of the person.

### Libraries Used :-
1. OpenCV
2. Numpy
3. OS
