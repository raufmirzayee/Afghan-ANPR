from ultralytics import YOLO
import unicodedata
import re
import cv2



def Dari_to_English(text):
    if text is None:
        return "" 

    Dari_to_En = {
        '1': '۱',
        '2': '۲',
        '3': '۳',
        '4': '۴',
        '5': '۵',
        '6': '۶',
        '7': '۷',
        '8': '۸',
        '9': '۹'
    }
    
    # replace Dari numbers with English numbers
    english_text = ''.join([Dari_to_En.get(char, char) for char in text])

    return english_text



def reg(text):

    # Decode Dari text to Unicode
    Dari_digits = unicodedata.normalize('NFC', text)

    # Regular Expression'
    matches = re.findall(r'[۱-۹١-٩]+', Dari_digits)

    # Find all matches 
    if matches:
        matches = "".join(matches)
        return matches


# Load model
license_plate_reader = YOLO('models/CNN.pt')
def read_license_plate(license_crop, class_names):
    char_detections = license_plate_reader(license_crop)[0]
    total_score = 0.0
    for char_detection in char_detections.boxes.data.tolist():
        x1, y1, x2, y2, score_class, class_id_class = char_detection
        try:    
            if class_names == 'Dari':
                detected_classes =  Sorting(char_detections)
                translated_final = [translate_class_label(label) for label in detected_classes]
                clean = ''
                for i in translated_final:
                    clean += i
                text = reg(clean)          
                if len(text) == 6:
                    text = text[1:]

                total_score += score_class
                return Dari_to_English(text), total_score
            
            elif class_names == 'Interval':
                translated_final = [translate_class_label(label) for label in detected_classes]
                text = reg(translated_final)
                total_score += score_class
                return text, total_score
            

            elif class_names == 'English':
                detected_classes =  Sorting(char_detections)
                translated_final = [translate_class_label(label) for label in detected_classes]
                clean = ''
                for i in translated_final:
                        clean += i
                # Use re.sub to replace the matched pattern with an empty string
                text = re.sub(r'[آ-ی-۱-۹A-Z]', '', clean)
                if len(text) == 6:
                    text = text[1:]  
                total_score += score_class

                return text, total_score
        except:
            pass
    return None, None

def Sorting(char_detection):
    detected_classes = []

    # Sort the boxes (left to right)
    sorted_boxes = sorted(char_detection.boxes, key=lambda box: box.xyxy[0, 0].item())

    #  sorted boxes
    for box in sorted_boxes:
        class_id = char_detection.names[box.cls[0].item()]  # Get labels
        cords = box.xyxy[0].tolist()  # Get bounding box 
        cords = [round(x) for x in cords]  # coordinates to integers
        detected_classes.append(class_id)  # Append class label to final list
    return detected_classes



def translate_class_label(class_label):
    translation_dict = {
        '1': '1',
        '2': '2',
        '3': '3',
        '4': '4',
        '5': '5',
        '6': '6',
        '7': '7',
        '8': '8',
        '9': '9',
        'eight': '۸',
        'five': '۵',
        'four': '۴',
        'nine': '۹',
        'one': '۱',
        'seven': '۷',
        'six': '۶',
        'three': '۳',
        'two': '۲'
    }

    return translation_dict.get(class_label, "")



def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=3, line_length_x=20, line_length_y=20):
  x1, y1 = top_left
  x2, y2 = bottom_right

  cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
  cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

  cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
  cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

  cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
  cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

  cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
  cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
  return img



# from easyocr import Reader
# reader = Reader(['en', 'fa'], gpu=False)

# def read_license_plate(license_crop, class_name):
#   detections = reader.readtext(license_crop)

#   for detection in detections:
#     bbox, text, score = detection
#     text = text.upper().replace(' ', '')

#     if class_name == 'Dari':
#       text =reg(text)
#       return Dari_to_English(text), score

#     elif class_name == 'English':
#       numbers = re.sub(r'[^0-9]', '', text)
#       return numbers, score

#     elif class_name == 'Interval':
#       text = reg(text)
#       return Dari_to_English(text), score

#   return None, None
