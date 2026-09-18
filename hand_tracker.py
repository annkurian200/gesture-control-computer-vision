import cv2
import mediapipe as mp
import numpy as np
import time


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)


# ============================================================
# START
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    cap = cv2.VideoCapture(0)

    success, frame = cap.read()

    if not success:
        print("Could not open camera.")
        exit()

    # Mirror camera
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape


    # ========================================================
    # DRAWING DATA
    # ========================================================

    # Every completed stroke is stored here
    strokes = []

    # Current stroke being drawn
    current_stroke = []

    # Previous finger position
    previous_point = None

    # Canvas
    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )


    # ========================================================
    # SMOOTHING
    # ========================================================

    smooth_x = None
    smooth_y = None


    # ========================================================
    # PINCH VARIABLES
    # ========================================================

    pinch_frames = 0
    not_pinch_frames = 0
    is_drawing = False


    # ========================================================
    # OPEN PALM VARIABLES
    # ========================================================

    open_palm_frames = 0
    clear_triggered = False


    # ========================================================
    # FIST / UNDO VARIABLES
    # ========================================================

    fist_frames = 0
    undo_triggered = False


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)


        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )


        # ----------------------------------------------------
        # HAND DETECTION
        # ----------------------------------------------------

        timestamp_ms = int(
            time.monotonic() * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # ====================================================
        # HAND FOUND
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # =================================================
            # INDEX FINGER POSITION
            # =================================================

            index_tip = hand[8]

            target_x = int(
                index_tip.x * width
            )

            target_y = int(
                index_tip.y * height
            )


            # =================================================
            # SMOOTH MOVEMENT
            # =================================================

            if smooth_x is None:

                smooth_x = target_x
                smooth_y = target_y

            else:

                smooth_x = int(
                    smooth_x * 0.6 +
                    target_x * 0.4
                )

                smooth_y = int(
                    smooth_y * 0.6 +
                    target_y * 0.4
                )


            x = smooth_x
            y = smooth_y


            # Tracking dot
            cv2.circle(
                frame,
                (x, y),
                10,
                (255, 0, 0),
                -1
            )


            # =================================================
            # PINCH DISTANCE
            # =================================================

            thumb_tip = hand[4]

            distance = (
                (thumb_tip.x - index_tip.x) ** 2 +
                (thumb_tip.y - index_tip.y) ** 2
            ) ** 0.5


            # =================================================
            # FINGER STATES
            # =================================================

            index_up = hand[8].y < hand[6].y
            middle_up = hand[12].y < hand[10].y
            ring_up = hand[16].y < hand[14].y
            pinky_up = hand[20].y < hand[18].y


            # =================================================
            # OPEN PALM
            # =================================================

            open_palm = (
                index_up and
                middle_up and
                ring_up and
                pinky_up and
                distance > 0.10
            )


            # =================================================
            # FIST
            # =================================================

            fist = (
                not index_up and
                not middle_up and
                not ring_up and
                not pinky_up
            )


            # =================================================
            # PRIORITY 1 — PINCH / DRAW
            # =================================================

            if distance < 0.08:

                pinch_frames += 1
                not_pinch_frames = 0

                open_palm_frames = 0
                clear_triggered = False

                fist_frames = 0
                undo_triggered = False


                if pinch_frames >= 2:

                    is_drawing = True

                    cv2.putText(
                        frame,
                        "PINCH - DRAWING",
                        (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )


                    # -----------------------------------------
                    # START NEW STROKE
                    # -----------------------------------------

                    if previous_point is None:

                        current_stroke = []

                        current_stroke.append(
                            (x, y)
                        )


                    # -----------------------------------------
                    # CONTINUE STROKE
                    # -----------------------------------------

                    else:

                        dx = x - previous_point[0]
                        dy = y - previous_point[1]

                        movement = (
                            dx ** 2 +
                            dy ** 2
                        ) ** 0.5


                        if movement < 80:

                            current_stroke.append(
                                (x, y)
                            )


                    previous_point = (x, y)


            # =================================================
            # PRIORITY 2 — OPEN PALM / CLEAR
            # =================================================

            elif open_palm:

                pinch_frames = 0
                not_pinch_frames = 0

                is_drawing = False

                previous_point = None

                current_stroke = []

                fist_frames = 0
                undo_triggered = False

                open_palm_frames += 1


                cv2.putText(
                    frame,
                    "OPEN PALM",
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )


                # Clear after holding palm
                if (
                    open_palm_frames >= 10
                    and not clear_triggered
                ):

                    strokes = []

                    canvas[:] = 0

                    clear_triggered = True


            # =================================================
            # PRIORITY 3 — FIST / UNDO
            # =================================================

            elif fist:

                pinch_frames = 0
                not_pinch_frames = 0

                is_drawing = False

                previous_point = None

                current_stroke = []

                open_palm_frames = 0
                clear_triggered = False


                fist_frames += 1


                cv2.putText(
                    frame,
                    "FIST - UNDO",
                    (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )


                # -----------------------------------------
                # UNDO ONCE AFTER HOLDING FIST
                # -----------------------------------------

                if (
                    fist_frames >= 18
                    and not undo_triggered
                ):

                    if len(strokes) > 0:

                        # Remove last stroke
                        strokes.pop()


                    # Rebuild canvas
                    canvas[:] = 0


                    for stroke in strokes:

                        if len(stroke) >= 2:

                            pts = np.array(
                                stroke,
                                dtype=np.int32
                            )

                            cv2.polylines(
                                canvas,
                                [pts],
                                False,
                                (255, 255, 255),
                                5,
                                cv2.LINE_AA
                            )


                    undo_triggered = True


            # =================================================
            # TEMPORARY PINCH LOSS
            # =================================================

            elif is_drawing:

                not_pinch_frames += 1

                # Allow a few missed frames
                if not_pinch_frames <= 5:

                    cv2.putText(
                        frame,
                        "DRAWING",
                        (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                else:

                    # Finish current stroke
                    if len(current_stroke) >= 2:

                        strokes.append(
                            current_stroke.copy()
                        )


                    current_stroke = []

                    is_drawing = False

                    previous_point = None

                    pinch_frames = 0


            # =================================================
            # PAUSED
            # =================================================

            else:

                pinch_frames = 0
                not_pinch_frames = 0

                open_palm_frames = 0
                clear_triggered = False

                fist_frames = 0
                undo_triggered = False

                previous_point = None

                cv2.putText(
                    frame,
                    "PAUSED",
                    (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )


            # =================================================
            # DRAW ALL SAVED STROKES
            # =================================================

            canvas[:] = 0


            # Saved strokes
            for stroke in strokes:

                if len(stroke) >= 2:

                    pts = np.array(
                        stroke,
                        dtype=np.int32
                    )

                    cv2.polylines(
                        canvas,
                        [pts],
                        False,
                        (255, 255, 255),
                        5,
                        cv2.LINE_AA
                    )


            # Current stroke
            if len(current_stroke) >= 2:

                pts = np.array(
                    current_stroke,
                    dtype=np.int32
                )

                cv2.polylines(
                    canvas,
                    [pts],
                    False,
                    (255, 255, 255),
                    5,
                    cv2.LINE_AA
                )


            # =================================================
            # LANDMARKS
            # =================================================

            for landmark in hand:

                lx = int(
                    landmark.x * width
                )

                ly = int(
                    landmark.y * height
                )

                cv2.circle(
                    frame,
                    (lx, ly),
                    5,
                    (0, 255, 0),
                    -1
                )


            cv2.putText(
                frame,
                "HAND DETECTED",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            smooth_x = None
            smooth_y = None

            pinch_frames = 0
            not_pinch_frames = 0

            open_palm_frames = 0
            clear_triggered = False

            fist_frames = 0
            undo_triggered = False

            is_drawing = False

            previous_point = None

            current_stroke = []


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "Hand Tracking",
            frame
        )

        cv2.imshow(
            "Canvas",
            canvas
        )


        # Q = quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    cv2.destroyAllWindows()