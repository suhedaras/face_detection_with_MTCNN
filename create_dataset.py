import cv2
import numpy as np
from facenet_pytorch import MTCNN
import os
from utils.settings import Settings

sett = Settings()


class FaceDetector(object):

    def __init__(self, mtcnn_input):
        self.control_dir()
        self.mtcnn = mtcnn_input
        self.project_absolute_path = sett.project_absolute_path
        self.source_path = sett.source_path

    def draw_frame(self, frame, boxes, probs, landmarks):

        self.frames = []

        try:
            for box, prob, ld in zip(boxes, probs, landmarks):
                box = np.int32(box)
                ld = np.int32(ld)
                self.frames.append(frame)

                # Draw face
                # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)

                # Show probability
                # cv2.putText(frame, str(prob), (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                #             cv2.LINE_AA)

                # Draw landmarks dot
                # cv2.circle(frame, tuple(ld[0]), 4, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[1]), 4, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[2]), 4, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[3]), 4, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[4]), 4, (0, 0, 255), -1)

                # Draw left_eye
                vis_resize = np.array([30, 20])
                left_eye = np.int32(ld[0])
                self.x_min0, self.y_min0 = left_eye - vis_resize
                self.x_max0, self.y_max0 = left_eye + vis_resize
                # cv2.rectangle(frame, (self.x_min0, self.y_min0), (self.x_max0, self.y_max0), (0, 255, 0), 2)

                # Draw right_eye
                vis_resize = np.array([30, 20])
                right_eye = np.int32(ld[1])
                self.x_min1, self.y_min1 = right_eye - vis_resize
                self.x_max1, self.y_max1 = right_eye + vis_resize
                # cv2.rectangle(frame, (self.x_min1, self.y_min1), (self.x_max1, self.y_max1), (0, 255, 0), 2)

                # Draw mouth
                vis_resize = np.array([75, 25])
                mouth_left = np.int32(ld[3])
                mouth_right = np.int32(ld[4])
                self.x_min2, self.y_min2 = mouth_right - vis_resize
                self.x_max2, self.y_max2 = mouth_left + vis_resize
                # cv2.rectangle(frame, (self.x_min2, self.y_min2), (self.x_max2, self.y_max2), (0, 255, 0), 2)

        except:
            pass

        return frame

    def control_dir(self):

        if not os.path.isdir(sett.outs_left_eye_path):
            os.makedirs(sett.outs_left_eye_path)
        if not os.path.isdir(sett.outs_right_eye_path):
            os.makedirs(sett.outs_right_eye_path)
        if not os.path.isdir(sett.outs_mouth_path):
            os.makedirs(sett.outs_mouth_path)
        if not os.path.isdir(sett.outs_face_path):
            os.makedirs(sett.outs_face_path)

    def create_dataset(self, frame, i):

        left_eye_crop = frame[self.y_min0: self.y_max0, self.x_min0: self.x_max0]
        image_label = os.path.join(sett.outs_left_eye_path, "image_" + str(i) + ".jpg")
        cv2.imwrite(image_label, left_eye_crop)

        right_eye_crop = frame[self.y_min1: self.y_max1, self.x_min1: self.x_max1]
        image_label2 = os.path.join(sett.outs_right_eye_path, "image_" + str(i) + ".jpg")
        cv2.imwrite(image_label2, right_eye_crop)

        mouth_crop = frame[self.y_min2: self.y_max2, self.x_min2: self.x_max2]
        image_label3 = os.path.join(sett.outs_mouth_path, "image_" + str(i) + ".jpg")
        cv2.imwrite(image_label3, mouth_crop)

        image_label_4 = [f'./result/face/image_{i}.jpg']
        for frame, path in zip(self.frames, image_label_4):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mtcnn(frame, save_path=path)

    def run(self):
        """
                   Run the FaceDetector and draw landmarks and boxes around detected faces, 0 for webcam
        """
        cap = cv2.VideoCapture(self.source_path)
        frame_count = 0
        i = 0
        while True:
            ret, frame = cap.read()
            try:
                if frame_count % 6 == 0:
                    # Detect face box, probability and landmarks
                    boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)

                    self.draw_frame(frame, boxes, probs, landmarks)
                    self.create_dataset(frame, i)
                    # cv2.imshow('Face Detection', frame)

                frame_count += 1
                i = i + 1
            except:
                break

            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Run the app
    mtcnn = MTCNN()
    fcd = FaceDetector(mtcnn)
    fcd.run()
