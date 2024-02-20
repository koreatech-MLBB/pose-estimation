# import cv2
# import mediapipe as mp
# import numpy as np
#
#
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
#
# pose_val = {
#   'NOSE': 0,
#   'LEFT_EYE_INNER': 1,
#   'LEFT_EYE': 2,
#   'LEFT_EYE_OUTER': 3,
#   'RIGHT_EYE_INNER': 4,
#   'RIGHT_EYE': 5,
#   'RIGHT_EYE_OUTER': 6,
#   'LEFT_EAR': 7,
#   'RIGHT_EAR': 8,
#   'MOUTH_LEFT': 9,
#   'MOUTH_RIGHT': 10,
#   'LEFT_SHOULDER': 11,
#   'RIGHT_SHOULDER': 12,
#   'LEFT_ELBOW': 13,
#   'RIGHT_ELBOW': 14,
#   'LEFT_WRIST': 15,
#   'RIGHT_WRIST': 16,
#   'LEFT_PINKY': 17,
#   'RIGHT_PINKY': 18,
#   'LEFT_INDEX': 19,
#   'RIGHT_INDEX': 20,
#   'LEFT_THUMB': 21,
#   'RIGHT_THUMB': 22,
#   'LEFT_HIP': 23,
#   'RIGHT_HIP': 24,
#   'LEFT_KNEE': 25,
#   'RIGHT_KNEE': 26,
#   'LEFT_ANKLE': 27,
#   'RIGHT_ANKLE': 28,
#   'LEFT_HEEL': 29,
#   'RIGHT_HEEL': 30,
#   'LEFT_FOOT_INDEX': 31,
#   'RIGHT_FOOT_INDEX': 32
# }
#
# def calculate_angle(a, b, c):
#     a = np.array(a)  # First
#     b = np.array(b)  # Mid
#     c = np.array(c)  # End
#
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
#
#     if angle > 180.0:
#         angle = 360 - angle
#
#     return angle
#
# def calculate_distance(a, b):
#     a = np.array(a)
#     b = np.array(b)
#     distance = np.linalg.norm(a - b)
#     return distance
#
#
# cap = cv2.VideoCapture(0)
#
# # Setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
#
#         # Recolor image to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#         image.flags.writeable = False
#
#         # Make detection
#         results = pose.process(image)
#
#         # Recolor back to BGR
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#         # Extract landmarks
#         try:
#             landmarks = results.pose_landmarks.landmark
#
#             # Get coordinates
#             coordinates = {}
#             for name, val in pose_val.items():
#                 coordinates[name] = [round(landmarks[pose_val[name]].x, 3), round(landmarks[pose_val[name]].y, 3)]
#
#             # Calculate angle
#             angle = calculate_angle(coordinates["LEFT_SHOULDER"], coordinates["LEFT_ELBOW"], coordinates["LEFT_WRIST"])
#             # for x, y in coordinates.items():
#             #     print(x, y)
#             input()
#
#             # Visualize angle
#             cv2.putText(image, str(angle),
#                         tuple(np.multiply(coordinates["LEFT_ELBOW"], [640, 480]).astype(int)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
#                         )
#         except Exception as e:
#             print(e)
#
#
#         # Render detections
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
#                                   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
#                                   )
#
#         cv2.imshow('Mediapipe Feed', image)
#
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
from PoseEstimation import PoseEstimation

pe = PoseEstimation(min_detection_confidence=0.5, min_tracking_confidence=0.5, camNum=0)
pe.run()