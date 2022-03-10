# import time
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from utils.settings import Settings

sett = Settings()


class FaceDetector(object):

    def __init__(self, mtcnn_input):
        self.mtcnn = mtcnn_input
        self.project_absolute_path = sett.project_absolute_path
        self.source_path = sett.source_path

    def draw_frame(self, frame, boxes, probs, landmarks):

        try:
            for box, prob, ld in zip(boxes, probs, landmarks):

                box = np.int32(box)
                ld = np.int32(ld)

                # Draw rectangle for face
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)
                cv2.putText(frame, str(prob), (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Draw landmarks dot
                cv2.circle(frame, tuple(ld[0]), 3, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[1]), 3, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[2]), 3, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[3]), 3, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[4]), 3, (0, 0, 255), -1)
                
        except:
            pass

        return frame

    def run(self):
        """
            Run the FaceDetector and draw landmarks and boxes around detected faces, 0 for webcam
        """

        cap = cv2.VideoCapture(self.source_path)
        
        while True:
            ret, frame = cap.read()

            try:
                # Detect face box, probability and landmarks
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                self.draw_frame(frame, boxes, probs, landmarks)
            except:
                pass
            
            # time.sleep(0.01)

            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def image_detect(self):

        frame = cv2.imread(sett.image_source)

        vis_width = int(frame.shape[1] * sett.scale)
        vis_height = int(frame.shape[0] * sett.scale)

        frame = cv2.resize(frame, (vis_width, vis_height))

        boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)

        self.draw_frame(frame, boxes, probs, landmarks)

        cv2.imshow('Face Detection', frame)

        k = cv2.waitKey(0) & 0xFF

        if k == ord("q"):
            cv2.destroyAllWindows()


if __name__ == '__main__':
    # Run the app
    mtcnn = MTCNN()
    fcd = FaceDetector(mtcnn)
    #For face detection in the image
    # fcd.image_detect()
    fcd.run()





