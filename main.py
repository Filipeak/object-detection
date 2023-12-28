from ultralytics import YOLO
import cv2
import face_recognition
import math
import numpy as np

CAM_WIDTH = 640
CAM_HEIGHT = 480


cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)


def run_face_recognition():
    RECTAGLE_COLOR_BGR = (0, 255, 0)
    TEXT_COLOR_BGR = (255, 0, 0)
    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

    IMAGE_DOWNSCALE_FACTOR = 4
    UNKNOWN_FACE_NAME = "Unknown"

    known_image = face_recognition.load_image_file("res/Face.jpg")
    known_image_encoding = face_recognition.face_encodings(known_image)[0]
    known_face_encodings = [known_image_encoding]
    known_face_names = ["Me"]

    while True:
        success, img = cap.read()

        small_frame = cv2.resize(
            img,
            (0, 0),
            fx=(1 / IMAGE_DOWNSCALE_FACTOR),
            fy=(1 / IMAGE_DOWNSCALE_FACTOR),
        )
        rgb_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )
            name = UNKNOWN_FACE_NAME
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= IMAGE_DOWNSCALE_FACTOR
            right *= IMAGE_DOWNSCALE_FACTOR
            bottom *= IMAGE_DOWNSCALE_FACTOR
            left *= IMAGE_DOWNSCALE_FACTOR

            cv2.rectangle(img, (left, top), (right, bottom), RECTAGLE_COLOR_BGR, 2)
            cv2.rectangle(
                img,
                (left, bottom - 35),
                (right, bottom),
                RECTAGLE_COLOR_BGR,
                cv2.FILLED,
            )
            cv2.putText(
                img, name, (left + 6, bottom - 6), TEXT_FONT, 1, TEXT_COLOR_BGR, 1
            )

        cv2.imshow("Video", img)

        if cv2.waitKey(1) == ord("q"):
            break


def run_object_detection():
    RECTAGLE_COLOR_BGR = (0, 255, 0)
    TEXT_COLOR_BGR = (0, 255, 0)
    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

    model = YOLO("res/yolo-Weights/yolov8n.pt")

    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]

                print(box.cls)

                x1, y1, x2, y2 = xyxy
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                confidence = math.ceil((conf * 100)) / 100
                classIndex = int(cls)
                className = model.names[classIndex]

                print(f"Box: {className} - {confidence}")

                cv2.rectangle(img, (x1, y1), (x2, y2), RECTAGLE_COLOR_BGR, 3)
                cv2.putText(img, className, [x1, y1], TEXT_FONT, 1, TEXT_COLOR_BGR, 2)

        cv2.imshow("Webcam", img)

        if cv2.waitKey(1) == ord("q"):
            break


run_face_recognition()
# run_object_detection()

cap.release()
cv2.destroyAllWindows()
