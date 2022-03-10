
import os
from pathlib import Path


class Settings:
    def __init__(self):
        self.scale = 1

        self.project_absolute_path = Path(os.path.abspath(""))
        self.source_path = os.path.join(self.project_absolute_path, "test_data", "normal-00" + "." + "mp4")
        self.image_source = os.path.join(self.project_absolute_path, "test_data", "test_data" + "." + "jpg")

        self.outs_left_eye_path = os.path.join(self.project_absolute_path, "result", "left_eye")
        self.outs_right_eye_path = os.path.join(self.project_absolute_path, "result", "right_eye")
        self.outs_mouth_path = os.path.join(self.project_absolute_path, "result", "mouth")
        self.outs_face_path = os.path.join(self.project_absolute_path, "result", "face")
