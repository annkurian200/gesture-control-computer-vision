import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

with HandLandmarker.create_from_options(options) as landmarker:

    cap = cv2.VideoCapture(0)
    success, frame = cap.read()
    height, width, _ = frame.shape 
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    previous_point = None
    smooth_x = None
    smooth_y = None
    while True:
        success, frame = cap.read()
        if not success:
            break

        # OpenCV uses BGR, MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Detect hands
        result = landmarker.detect(mp_image)

        # Draw hand landmarks
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                index_tip = hand[8]

                target_x = int(index_tip.x * frame.shape[1])
                target_y = int(index_tip.y * frame.shape[0])

                if smooth_x is None:
                    smooth_x = target_x
                    smooth_y = target_y

                smooth_x = int(smooth_x * 0.7 + target_x * 0.3)
                smooth_y = int(smooth_y * 0.7 + target_y * 0.3)
                x = smooth_x
                y = smooth_y
                cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)

                thumb_tip = hand[4]

                distance = ((thumb_tip.x - index_tip.x) ** 2 +
                            (thumb_tip.y - index_tip.y) ** 2) ** 0.5
                if distance < 0.05:
                    cv2.putText(
                        frame,
                        "PINCH",
                        (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                if hand[8].y < hand[6].y :
                    cv2.putText(
                frame,
                "ONE FINGER",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
                    )
                if hand[8].y < hand[6].y and hand[12].y < hand[10].y:
                    cv2.putText(
        frame,
        "TWO FINGERS",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
                    )
                if (hand[8].y < hand[6].y and
    hand[12].y < hand[10].y and
    hand[16].y < hand[14].y and
    hand[20].y < hand[18].y):

                    cv2.putText(
        frame,
        "OPEN PALM",
        (50, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
                    )
                if (hand[8].y > hand[5].y and
    hand[12].y > hand[9].y and
    hand[16].y > hand[13].y and
    hand[20].y > hand[17].y):

                    cv2.putText(
        frame,
        "FIST",
        (50, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
                    )
                if distance < 0.08:
                    cv2.putText(
                        frame,
                        "DRAWING",
                        (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                    if previous_point is not None:

                        distance_between = np.linalg.norm(
                            np.array((x, y)) - np.array(previous_point)
                        )

                        steps = max(1, int(distance_between / 5))

                        for i in range(1, steps + 1):
                            intermediate_x = int(
                            previous_point[0] +
                            (x - previous_point[0]) * i / steps
                            )

                            intermediate_y = int(
                            previous_point[1] +
                            (y - previous_point[1]) * i / steps
                            )

                            cv2.line(
                            canvas,
                            previous_point,
                            (intermediate_x, intermediate_y),
                            (255, 255, 255),
                            5
                            )

                    previous_point = (x, y)

                else:
                    cv2.putText(
                        frame,
                        "PAUSED",
                        (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                    
                    
                for landmark in hand:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(
    frame,
    "HAND DETECTED",
    (50, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
            )
        cv2.imshow("Hand Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cv2.imshow("Canvas", canvas)


    cap.release()
    cv2.destroyAllWindows()