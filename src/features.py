
import numpy as np

# Connections between the 21 hand landmarks.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def extract_hand_features(hand_landmarks):
    """
    Extract 126 features:
    2 hands x 21 landmarks x 3 coordinates.
    """

    hands = np.zeros((2, 21, 3), dtype=np.float32)

    if hand_landmarks:
        detected = []

        for hand in hand_landmarks:
            points = np.array(
                [[p.x, p.y, p.z] for p in hand],
                dtype=np.float32
            )

            # Normalize relative to wrist.
            points = points - points[0]

            scale = np.max(np.linalg.norm(points, axis=1))

            if scale > 1e-6:
                points = points / scale

            detected.append(points)

        # Keep a consistent left-to-right order.
        detected.sort(key=lambda p: p[0, 0])

        for idx, points in enumerate(detected[:2]):
            hands[idx] = points

    return hands.reshape(-1)


def draw_hand_landmarks(frame, hand_landmarks):
    """Draw detected hand landmarks using OpenCV."""

    import cv2

    height, width = frame.shape[:2]

    for hand in hand_landmarks or []:
        coordinates = []

        for point in hand:
            x = int(point.x * width)
            y = int(point.y * height)
            coordinates.append((x, y))

        # Draw hand connections.
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                coordinates[start],
                coordinates[end],
                (0, 255, 0),
                2
            )

        # Draw landmark points.
        for x, y in coordinates:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    return frame
