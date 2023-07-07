import sys
import copy
import csv
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
    """
    Initialize and return Mediapipe FaceMesh Solution Graph object
    """

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return face_mesh


def distance(point_1, point_2):
    """
    Calculate l2-norm between two points

    :param point_1:
    :param point_2:
    :return: L2-norm distance
    :rtype: tuple
    """

    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Eye Aspect Ratio for one eye.

    :param list landmarks: Detected landmarks list
    :param list refer_idxs: Index positions of the chosen landmarks
                            in order P1, P2, P3, P4, P5, P6
    :param int frame_width: Width of captured frame
    :param int frame_height: Height of captured frame

    :return: ear: (float) Eye aspect ratio;
             lm_coordinates: (tuple) Landmarks coordinate points
    :rtype: tuple
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


def infer_openvino_model(input_image, net, input_name, output_name):
    """
    Parent function for model inference.

    :param np.array input_image: Frame as numpy array
    :param IENetwork net: OpenVINO IENetwork model
    :param str input_name: Input layer name
    :param str output_name: Output layer name

    :return: Inference results (model output)
    :rtype: list
    """

    # Preprocess the image
    input_shape = net.input_info[input_name].input_data.shape
    resized_image = cv2.resize(input_image, (input_shape[3], input_shape[2]))
    preprocessed_image = resized_image.transpose((2, 0, 1))
    preprocessed_image = preprocessed_image.reshape(input_shape)

    # Run inference on the image
    exec_net = IE.load_network(network=net, device_name='CPU', num_requests=1)
    inference_results = exec_net.infer(inputs={input_name: preprocessed_image})[output_name][0]

    return inference_results


def get_head_pos(input_image, net, input_name, output_name):
    """
    Calculate head position (yaw).

    :param np.array input_image: Frame as numpy array
    :param IENetwork net: OpenVINO IENetwork model
    :param str input_name: Input layer name
    :param str output_name: Output layer name

    :return: Head position (yaw)
    :rtype: float
    """

    inference_results = infer_openvino_model(input_image, net, input_name, output_name)

    # Extract the output probabilities
    yaw_prob = inference_results[0]
    return abs(yaw_prob)


def get_drowsiness_index(input_image, net, input_name, output_name):
    """
    Calculate drowsiness index.

    :param np.array input_image: Frame as numpy array
    :param IENetwork net: OpenVINO IENetwork model
    :param str input_name: Input layer name
    :param str output_name: Output layer name

    :return: Drowsiness index
    :rtype: float
    """

    inference_results = infer_openvino_model(input_image, net, input_name, output_name)
    return inference_results[1]


def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    """
    Calculate average Eye Aspect Ratio for both eyes.

    :param list landmarks: Detected landmarks list
    :param list left_eye_idxs: Index positions of the chosen landmarks for left eye
                               in order P1, P2, P3, P4, P5, P6
    :param list right_eye_idxs: Index positions of the chosen landmarks for right eye
                                in order P1, P2, P3, P4, P5, P6
    :param int image_w: Width of captured frame
    :param int image_h: Height of captured frame

    :return: avg_ear (float) Average EAR for both left & right eye;
             lm_coordinates: (tuple) Landmarks coordinate points (left & right eye)
    :rtype: tuple
    """
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


def dump_stats(state_tracker: dict):
    """
    Dump current state into a .csv log file
    :param dict state_tracker: current state tracker dict
    :return: None
    """
    fieldnames = ['MINUTES_ELAPSED', 'EAR.average_triggered_time', 'EAR.triggers_per_minute',
                  'HEAD.average_triggered_time', 'HEAD.triggers_per_minute']
    file_exists = Path('stats.csv').is_file()

    with open('stats.csv', 'a' if file_exists else 'w+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'MINUTES_ELAPSED': state_tracker['MINUTES_ELAPSED'],
            'EAR.average_triggered_time': state_tracker['EAR']['average_triggered_time'],
            'EAR.triggers_per_minute': state_tracker['EAR']['triggers_per_minute'],
            'HEAD.average_triggered_time': state_tracker['HEAD']['average_triggered_time'],
            'HEAD.triggers_per_minute': state_tracker['HEAD']['triggers_per_minute']
        })


class VideoFrameHandler:
    def __init__(self, sys_config):
        """
        Initialize the necessary constants, mediapipe app
        and tracker variables
        """
        # General system configuration with constants
        self.sys_config = sys_config.properties
        self.root_dir = Path(Path(__file__).resolve().parent.parent.parent)

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
            min_tracking_confidence=self.sys_config['DETECTION']['MEDIAPIPE']['MIN_FACE_TRACKING_CONFIDENCE'])

        # OpenVINO head position and drowsiness detection models
        self.head_net = IE.read_network(
            model=self.root_dir / self.sys_config['DETECTION']['OPENVINO']['HEAD']['IR'],
            weights=self.root_dir / self.sys_config['DETECTION']['OPENVINO']['HEAD']['WEIGHTS'])
        self.drowsy_net = IE.read_network(
            model=self.root_dir / self.sys_config['DETECTION']['OPENVINO']['DROWSY']['IR'],
            weights=self.root_dir / self.sys_config['DETECTION']['OPENVINO']['DROWSY']['WEIGHTS'])

        # Tracking counters and sharing states in and out of callbacks
        self.default_entry = {
            "start_time": time.perf_counter(),
            "triggered_time": 0.0,
            "text_color": self.GREEN,
            "trigger_counter": 0,
            "average_triggered_time": 0.0,
            "triggers_per_minute": 0
        }

        self.state_tracker = {
            "FRAME_COUNTER": 0,
            "MINUTES_ELAPSED": 0,
            "CURRENT_TIME": time.time(),
            "EAR": {
                **copy.deepcopy(self.default_entry)
            },
            "HEAD": {
                **copy.deepcopy(self.default_entry),
                "position": 0.0,
            },
            "DROWSY": {
                **copy.deepcopy(self.default_entry),
                "is_drowsy": False
            },
            "play_alarm": False
        }

    def check_metric_trigger(self, value, threshold, state_value_label, wait_time, check_overcome=False):
        localtime = time.time()
        if localtime - self.state_tracker["CURRENT_TIME"] >= 60:
            self.state_tracker['MINUTES_ELAPSED'] += 1
            if self.state_tracker[state_value_label]['triggers_per_minute'] > 0:
                self.state_tracker[state_value_label]['triggers_per_minute'] = (self.state_tracker[state_value_label][
                                                                                    'triggers_per_minute'] +
                                                                                self.state_tracker[state_value_label][
                                                                                    "trigger_counter"]) / 2
            else:
                self.state_tracker[state_value_label]['triggers_per_minute'] = self.state_tracker[state_value_label][
                    "trigger_counter"]
            LOGGER.warning(f"Triggers by last 60 sec: {self.state_tracker[state_value_label]['trigger_counter']}")
            LOGGER.warning(f"Avg triggers by minute: {self.state_tracker[state_value_label]['triggers_per_minute']}")
            dump_stats(self.state_tracker)
            self.state_tracker[state_value_label]["trigger_counter"] = 0
            self.state_tracker["CURRENT_TIME"] = time.time()
        if (check_overcome and (value > threshold)) or (not check_overcome and (value < threshold)):
            current_time = time.perf_counter()
            self.state_tracker[state_value_label]['triggered_time'] += current_time - \
                                                                       self.state_tracker[state_value_label][
                                                                           'start_time']
            self.state_tracker[state_value_label]['start_time'] = current_time
            self.state_tracker[state_value_label]['text_color'] = self.RED
            self.state_tracker[state_value_label]["trigger_counter"] += 1

            if self.state_tracker[state_value_label]['triggered_time'] >= wait_time:
                self.state_tracker["play_alarm"] = True
                return True
        else:
            self.state_tracker[state_value_label]['start_time'] = time.perf_counter()
            self.state_tracker[state_value_label]['triggered_time'] = 0.0
            self.state_tracker[state_value_label]['text_color'] = self.GREEN
            self.state_tracker["play_alarm"] = False

            if self.state_tracker[state_value_label]['triggered_time'] > 0.0:
                self.state_tracker[state_value_label]['average_triggered_time'] = \
                    (self.state_tracker[state_value_label]['average_triggered_time'] +
                     self.state_tracker[state_value_label]['triggered_time']) / 2
            return False

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

        # processing: head position (yaw)
        head_conf = self.sys_config['DETECTION']['OPENVINO']['HEAD']
        head_pos = get_head_pos(frame, self.head_net, head_conf['INPUT_NAME'], head_conf['OUTPUT_NAME'])
        self.state_tracker['HEAD']['position'] = round(head_pos, 2)

        # processing: eyes aspect ratio
        results = self.facemesh_model.process(frame)

        # processing DL (each 10th frame for optimization), preparing output
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            EAR, coordinates = calculate_avg_ear(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], frame_w,
                                                 frame_h)
            drowsy_index = get_drowsiness_index(frame, self.drowsy_net,
                                                self.sys_config['DETECTION']['OPENVINO']['DROWSY']['INPUT_NAME'],
                                                self.sys_config['DETECTION']['OPENVINO']['DROWSY']['OUTPUT_NAME'])
            frame = plot_eye_landmarks(frame, coordinates[0], coordinates[1], self.state_tracker['EAR']['text_color'])

            if self.check_metric_trigger(EAR, thresholds['EAR_THRESH'], 'EAR', thresholds['WAIT_TIME']):
                plot_text(frame, "OPEN YOUR EYES!", ALM_txt_pos, self.state_tracker['EAR']['text_color'])
            if self.check_metric_trigger(head_pos, thresholds['HEAD_THRESH'], 'HEAD', thresholds['WAIT_TIME'], True):
                plot_text(frame, "EYES ON ROAD!", ALM_txt_pos, self.state_tracker['HEAD']['text_color'])
            if self.state_tracker['FRAME_COUNTER'] % 10 == 0:
                if self.check_metric_trigger(drowsy_index, thresholds['DROWSY_THRESH'], 'DROWSY',
                                             thresholds['WAIT_TIME'], True):
                    self.state_tracker['DROWSY']['is_drowsy'] = True
                else:
                    self.state_tracker['DROWSY']['is_drowsy'] = False
                self.state_tracker['FRAME_COUNTER'] = 0

            EAR_txt = f"EAR: {round(EAR, 2)}"
            CLOSED_TIME_txt = f"EYES CLOSED: {round(self.state_tracker['EAR']['triggered_time'], 3)} Secs"
            HEAD_text = f"HEAD POS: {round(self.state_tracker['HEAD']['position'], 3)}"
            STATE_text = f"STATE: {'DROWSY' if self.state_tracker['DROWSY']['is_drowsy'] else 'OK'}"

            plot_text(frame, EAR_txt, self.EAR_txt_pos, self.state_tracker['EAR']['text_color'])
            plot_text(frame, CLOSED_TIME_txt, CLOSED_TIME_txt_pos, self.state_tracker['EAR']['text_color'])
            plot_text(frame, HEAD_text, self.HEAD_txt_pos, self.state_tracker['HEAD']['text_color'])
            plot_text(frame, STATE_text, self.STATE_txt_pos, self.state_tracker['DROWSY']['text_color'])

        else:
            self.state_tracker['EAR']['start_time'] = time.perf_counter()
            self.state_tracker['HEAD']['start_time'] = time.perf_counter()
            self.state_tracker['DROWSY']['start_time'] = time.perf_counter()

            self.state_tracker['EAR']['triggered_time'] = 0.0
            self.state_tracker['HEAD']['triggered_time'] = 0.0
            self.state_tracker['DROWSY']['triggered_time'] = 0.0

            self.state_tracker['EAR']["text_color"] = self.GREEN
            self.state_tracker['HEAD']["text_color"] = self.GREEN
            self.state_tracker['DROWSY']["text_color"] = self.GREEN
            self.state_tracker["play_alarm"] = False

            # Flip the frame horizontally for a selfie-view display.
            frame = cv2.flip(frame, 1)

        return frame, self.state_tracker["play_alarm"]
