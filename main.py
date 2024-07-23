import os
import numpy as np
import cv2
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime
from util import read_license_plate, draw_border
from ultralytics import YOLO

cred = credentials.Certificate("tollsdb.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://tollsdb-a0b14-default-rtdb.firebaseio.com/",
    'storageBucket': "tollsdb-a0b14.appspot.com"
})

# instance
bucket = storage.bucket()

# Load models
license_plate_detector = YOLO('./models/best1.pt')
categorization_model = YOLO('./models/best2.pt') 


# Dictionary to store results
results = {}
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Importing the mode images into a list

imgBackground = cv2.imread('Resources/background.png')
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))


modeType = 0
counter = 0

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

  # Detect license plates
    license_plates = license_plate_detector(img)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Use the categorization model to categorize the license plate
        license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]
        license_plate_category = categorization_model(license_plate_crop)[0]

        # Get the recognized text from OCR results
        for categories_detection in license_plate_category.boxes.data.tolist():
            x1_class, y1_class, x2_class, y2_class, score_class, class_id_class = categories_detection

            if class_id != 0:
                # Crop license plate
                license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]

                # Crop the detected class from the license plate image
                class_name = None

                if class_id_class == 0:
                    class_name = "Dari"
                elif class_id_class == 1:
                    class_name = "English"
                elif class_id_class == 2:
                    class_name = "Interval"

                if class_name:
                    class_crop = license_plate_crop[int(y1_class):int(y2_class), int(x1_class):int(x2_class), :]
                    
                    # # Convert the class_crop to grayscale
                    # class_crop_gray = cv2.cvtColor(class_crop, cv2.COLOR_BGR2GRAY)

                    # # Apply adaptive thresholding
                    # _, class_crop_thresh = cv2.threshold(class_crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # # Store the cropped class and its grayscale version in the dictionaries
                    # class_crops[class_name] = class_crop
                    # class_crops_gray[class_name] = class_crop_thresh                   

                    # Perform OCR on the segmented license plate using the recognized language
                    
                    class_text, class_text_score = read_license_plate(class_crop, class_name)
                    print(class_text)
                    if class_text is not None:
                        # Check if the car_id is already in the results dictionary
                        results = {'license_plate': {'bbox': [x1, y1, x2, y2],
                                                            'bbox_score': score,
                                                            'Dari_Text': '',
                                                            'Interval_Text': '',
                                                            'English_Text': ''}}

                            # Update the text values in the results dictionary
                        results['license_plate'][f'{class_name}_Text'] = class_text

                if class_text is not None:
                    x1, y1, x2, y2, score, class_id = license_plate
                    imgBackground = draw_border(imgBackground, (int(x1+50), int(y1+168)), (int(x2+50), int(y2+168)), (0, 255, 0))

                    id = class_text
                    if counter == 0:
                        cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                        cv2.imshow("TOLLS Collection", imgBackground)
                        cv2.waitKey(1)
                        counter = 1
                        modeType = 1

            try:
                if counter != 0:
                    if counter == 1:
                        # Get Data
                        driverInfo = db.reference(f'Drivers/{id}').get()
                        # Get Image 
                        blob = bucket.get_blob(f'Images/{id}.png')
                        if blob is not None:
                            # Get Image from storage
                            blob = bucket.get_blob(f'Images/{id}.png')
                            array = np.frombuffer(blob.download_as_string(), np.uint8)
                            img = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                            # Update data of attendance
                            datetimeObject = datetime.strptime(driverInfo['last_attendance_time'],
                                                            "%Y-%m-%d %H:%M:%S")
                            secondsElapsed = (datetime.now() - datetimeObject).total_seconds()

                            print(secondsElapsed)
                            
                            if secondsElapsed > 30:
                                ref = db.reference(f'Drivers/{id}')
                                driverInfo['total_attendance'] += 1
                                ref.child('total_attendance').set(driverInfo['total_attendance'])
                                ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                            else:
                                modeType = 3
                                counter = 0
                                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                    if modeType != 3:
                        if 10 < counter < 20:
                            modeType = 2
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                        if counter <= 10:
                            cv2.putText(imgBackground, str(driverInfo['total_attendance']), (861, 125),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                            cv2.putText(imgBackground, str(driverInfo['Route']), (1006, 550),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(imgBackground, str(id), (1006, 493),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(imgBackground, str(driverInfo['Car_Type']), (910, 625),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                            cv2.putText(imgBackground, str(driverInfo['Tolls_amount']), (1045, 625),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                            cv2.putText(imgBackground, str(driverInfo['starting_year']), (1160, 625),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                            (w, h), _ = cv2.getTextSize(driverInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                            offset = (414 - w) // 2
                            cv2.putText(imgBackground, str(driverInfo['name']), (808 + offset, 445),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)
                            try:
                                imgBackground[175:175 + 216, 909:909 + 216] = imgdriver
                            except:
                                pass
                        counter += 1

                        if counter >= 20:
                            counter = 0
                            modeType = 0
                            driverInfo = []
                            imgdriver = []
                            imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
            except:
                pass
        else:
            modeType = 0
            counter = 0

    cv2.imshow("TOLLS Collection", imgBackground)
    cv2.waitKey(1)
