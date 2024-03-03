import cv2
import numpy as np
from ultralytics import YOLO
from keras.models import load_model


model_segment = YOLO('yolov8s-seg.pt')
model_detect = YOLO('yolov8s-detect-landmark.pt')
model_recognition = load_model('modelInception_final.h5')

labels = ['Bolshoi Theatre', 'Cathedral of Christ the Saviour', 'Hilton Moscow Leningradskaya', 'Hotel Ucraina',
          'Kotelnicheskaya Embankment Building', 'Kudrinskaya Square Building', 'Lomonosov Moscow State University', 'Moscow City',
          'Red Gate Building', 'Russian State Library', 'St. Basils Cathedral', 'State Historical Museum', 'The Ministry of Foreign Affairs of Russia']


def get_image(image_path):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb


def bounding_human(image, results):
    masks = []

    for result in results:
        masks_data = result.masks
        boxes = result.boxes.cpu().numpy()
        classes = boxes.cls

        # Check to see if you can find a person or any object
        if masks_data is None:
            return image

        # Check if the mask is human or not
        for index in range(len(masks_data)):
            if classes[index] == 0:
                masks.append(np.int32(masks_data[index].xy))

    for mask in masks:
        cv2.fillPoly(image, [mask], (0, 0, 0))

    return image


def box_building(image, results):
    boxes = []
    for result in results:
        boxes_data = result.boxes.cpu().numpy()

        # Check to see if you can find any building
        if boxes_data is None:
            return image

        for box in boxes_data:
            boxes.append(np.int32(box.xyxy[0]))

    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    return image, boxes


def crop_image(image, boxes):
    images_cropped = []
    for box in boxes:
        x_min = box[0]
        y_min = box[1]
        x_max = box[2]
        y_max = box[3]
        image_crop = image[y_min:y_max, x_min:x_max]
        images_cropped.append(image_crop)

    return images_cropped


def recognition(images):
    labels_predict = []
    probs_predict = []
    for image in images:
        # Data normalization
        img_normalized = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_normalized = cv2.resize(img_normalized, (150, 150))
        x = np.expand_dims(img_normalized, axis=0)
        img = np.vstack([x])

        # Recognition
        classes = model_recognition.predict(img, batch_size=10)
        print(classes)
        print('sum:', np.sum(classes))

        prob_predict = np.max(classes)
        index_max = np.argmax(classes)
        label_predict = labels[index_max] if classes[0][index_max] > 0.5 else 'Unknown'
        labels_predict.append(label_predict)
        probs_predict.append(prob_predict)
        print(f"index max: {index_max}")
        print(f"label predict: {label_predict}")

    print(labels_predict)
    print(probs_predict)
    label_max = labels_predict[np.argmax(probs_predict)]

    return label_max


def recognize(image):
    # results_segment = model_segment(image)
    # image_without_human = image.copy()
    # image_without_human = bounding_human(image_without_human, results_segment)

    # results_building = model_detect(image_without_human)
    results_building = model_detect(image)
    # image_with_box_building = image_without_human.copy()
    image_with_box_building = image.copy()
    image_with_box_building, boxes = box_building(image_with_box_building, results_building)
    if not bool(boxes):
        return 'unknown'

    # img_src = image.copy()
    # results_withImgSrc = model_detect(img_src)
    # imgSrc_with_box_building, boxes_src = box_building(img_src, results_withImgSrc)

    # images_cropped = crop_image(image_without_human, boxes)
    images_cropped = crop_image(image, boxes)

    label_predict = recognition(images_cropped)

    return label_predict
