import sys
import time
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
from openvino.inference_engine import IECore

# to be able to access packages from any script workdir
sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.logger_conf import configure_logger

LOGGER = configure_logger(__name__)
IE = IECore()


def get_mediapipe_app(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
):
    """Initialize and return Mediapipe FaceMesh Solution Graph object"""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return face_mesh


def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Eye Aspect Ratio for one eye.

    Args:
        landmarks: (list) Detected landmarks list
        refer_idxs: (list) Index positions of the chosen landmarks
                            in order P1, P2, P3, P4, P5, P6
        frame_width: (int) Width of captured frame
        frame_height: (int) Height of captured frame

    Returns:
        ear: (float) Eye aspect ratio
    """
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points


def get_head_pos(input_image, net):
    # Preprocess the image
    input_shape = net.input_info['data'].input_data.shape
    resized_image = cv2.resize(input_image, (input_shape[3], input_shape[2]))
    preprocessed_image = resized_image.transpose((2, 0, 1))
    preprocessed_image = preprocessed_image.reshape(input_shape)

    # Run inference on the image
    exec_net = IE.load_network(network=net, device_name='CPU', num_requests=1)
    inference_results = exec_net.infer(inputs={'data': preprocessed_image})

    # Extract the output probabilities
    yaw_prob = inference_results['angle_y_fc'][0][0]
    return abs(yaw_prob)


def check_drowsiness(input_image, net):
    # Preprocess the image
    input_shape = net.input_info['input.1'].input_data.shape
    resized_image = cv2.resize(input_image, (input_shape[3], input_shape[2]))
    preprocessed_image = resized_image.transpose((2, 0, 1))
    preprocessed_image = preprocessed_image.reshape(input_shape)

    # Run inference on the image
    exec_net = IE.load_network(network=net, device_name='CPU', num_requests=1)
    inference_results = exec_net.infer(inputs={'input.1': preprocessed_image})['519'][0]

    threshold = 10
    return inference_results[1] > threshold


def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    # Calculate Eye Aspect Ratio (EAR)

    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)


def plot_eye_landmarks(frame, left_lm_coordinates, right_lm_coordinates, color):
    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(frame, coord, 2, color, -1)

    frame = cv2.flip(frame, 1)
    return frame


def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image


class VideoFrameHandler:
    def __init__(self, sys_config):
        """
        Initialize the necessary constants, mediapipe app
        and tracker variables
        """
        # General system configuration with constants
        self.sys_config = sys_config.properties

        # Left and right eye chosen landmarks
        self.eye_idxs = {
            "left": self.sys_config['DETECTION']['EAR']['LANDMARKS']['LEFT'],
            "right": self.sys_config['DETECTION']['EAR']['LANDMARKS']['RIGHT'],
        }

        # Used for coloring landmark points. Value depends on the current EAR value
        self.RED = (0, 0, 255)  # BGR
        self.GREEN = (0, 255, 0)  # BGR

        # Positions for text overlays
        self.EAR_txt_pos = (10, 30)
        self.HEAD_txt_pos = (10, 50)
        self.STATE_txt_pos = (10, 75)

        # MediaPipe face detection pipeline
        self.facemesh_model = get_mediapipe_app(
            max_num_faces=self.sys_config['DETECTION']['MEDIAPIPE']['MAX_FACES'],
            min_detection_confidence=self.sys_config['DETECTION']['MEDIAPIPE']['MIN_FACE_DETECTION_CONFIDENCE'],
            min_tracking_confidence=self.sys_config['DETECTION']['MEDIAPIPE']['MIN_FACE_TRACKING_CONFIDENCE']
        )

        # OpenVINO head position and drowsiness detection models
        self.head_net = IE.read_network(model=self.sys_config['DETECTION']['OPENVINO']['HEAD_POS_MODEL'],
                                        weights=self.sys_config['DETECTION']['OPENVINO']['HEAD_POS_MODEL_WEIGHTS'])
        self.drowsy_net = IE.read_network(model=self.sys_config['DETECTION']['OPENVINO']['DROWSY_MODEL'],
                                          weights=self.sys_config['DETECTION']['OPENVINO']['DROWSY_MODEL_WEIGHTS'])

        # Tracking counters and sharing states in and out of callbacks
        self.state_tracker = {
            "FRAME_COUNTER": 0,
            "EAR_start_time": time.perf_counter(),
            "CLOSED_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
            "EAR_COLOR": self.GREEN,
            "play_alarm": False,
            "HEAD_start_time": time.perf_counter(),
            "HEAD_POS": 0.0,
            "HEAD_MOVED_TIME": 0.0,  # Holds the amount of time passed with HEAD_POS < HEAD_THRESH
            "HEAD_POS_COLOR": self.GREEN,
            "IS_DROWSY": False
        }

    def process(self, frame: np.array, thresholds: dict):
        """
        This function is used to implement Drowsy detection algorithm

        Args:
            frame: (np.array) Input frame matrix.
            thresholds: (dict) Contains threshold values

        Returns:
            The processed frame and a boolean flag to
            indicate if the alarm should be played or not.
        """

        self.state_tracker['FRAME_COUNTER'] += 1

        # To improve performance,
        # mark the frame as not writeable to pass by reference.
        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape

        CLOSED_TIME_txt_pos = (10, int(frame_h // 2 * 1.7))
        ALM_txt_pos = (10, int(frame_h // 2 * 1.85))

        results = self.facemesh_model.process(frame)
        head_pos = get_head_pos(frame, self.head_net)

        self.state_tracker["HEAD_POS"] = round(head_pos, 2)

        # processing
        if self.state_tracker['FRAME_COUNTER'] % 10 == 0:
            self.state_tracker['IS_DROWSY'] = check_drowsiness(frame, self.drowsy_net)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            EAR, coordinates = calculate_avg_ear(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], frame_w,
                                                 frame_h)
            frame = plot_eye_landmarks(frame, coordinates[0], coordinates[1], self.state_tracker["EAR_COLOR"])

            if EAR < thresholds["EAR_THRESH"]:
                current_time = time.perf_counter()

                self.state_tracker["CLOSED_TIME"] += current_time - self.state_tracker["EAR_start_time"]
                self.state_tracker["EAR_start_time"] = current_time
                self.state_tracker["EAR_COLOR"] = self.RED

                if self.state_tracker["CLOSED_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_alarm"] = True
                    plot_text(frame, "OPEN YOUR EYES!", ALM_txt_pos, self.state_tracker["EAR_COLOR"])
            else:
                self.state_tracker["EAR_start_time"] = time.perf_counter()
                self.state_tracker["CLOSED_TIME"] = 0.0
                self.state_tracker["EAR_COLOR"] = self.GREEN
                self.state_tracker["play_alarm"] = False

            if head_pos > thresholds["HEAD_THRESH"]:
                current_time = time.perf_counter()

                self.state_tracker["HEAD_MOVED_TIME"] += current_time - self.state_tracker["HEAD_start_time"]
                self.state_tracker["HEAD_start_time"] = current_time
                self.state_tracker["HEAD_POS_COLOR"] = self.RED

                if self.state_tracker["HEAD_MOVED_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_alarm"] = True
                    plot_text(frame, "EYES ON ROAD!", ALM_txt_pos, self.state_tracker["HEAD_POS_COLOR"])
            else:
                self.state_tracker["HEAD_start_time"] = time.perf_counter()
                self.state_tracker["HEAD_MOVED_TIME"] = 0.0
                self.state_tracker["HEAD_POS_COLOR"] = self.GREEN
                self.state_tracker["play_alarm"] = False

            EAR_txt = f"EAR: {round(EAR, 2)}"
            CLOSED_TIME_txt = f"DROWSY: {round(self.state_tracker['CLOSED_TIME'], 3)} Secs"
            HEAD_text = f"HEAD POS: {self.state_tracker['HEAD_POS']}"
            STATE_text = f"STATE: {'DROWSY' if self.state_tracker['IS_DROWSY'] else 'OK'}"

            plot_text(frame, EAR_txt, self.EAR_txt_pos, self.state_tracker["EAR_COLOR"])
            plot_text(frame, CLOSED_TIME_txt, CLOSED_TIME_txt_pos, self.state_tracker["EAR_COLOR"])
            plot_text(frame, HEAD_text, self.HEAD_txt_pos, self.state_tracker["HEAD_POS_COLOR"])
            plot_text(frame, STATE_text, self.STATE_txt_pos, self.state_tracker["HEAD_POS_COLOR"])  # TODO: change

        else:
            self.state_tracker["EAR_start_time"] = time.perf_counter()
            self.state_tracker["HEAD_start_time"] = time.perf_counter()
            self.state_tracker["CLOSED_TIME"] = 0.0
            self.state_tracker["EAR_COLOR"] = self.GREEN
            self.state_tracker["play_alarm"] = False

            # Flip the frame horizontally for a selfie-view display.
            frame = cv2.flip(frame, 1)

        return frame, self.state_tracker["play_alarm"]
