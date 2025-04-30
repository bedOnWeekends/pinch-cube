import cv2
import mediapipe as mp
import math
import time
import logging
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hand_tracking.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Cube:
    def __init__(self):
        self.pieces = []

        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    if not (x == 0 and y == 0 and z == 0):
                        self.pieces.append([x, y, z, 0])

        self.colors = [
            (1, 0, 0),
            (1, 0.5, 0),
            (1, 1, 0),
            (1, 1, 1),
            (0, 0, 1),
            (0, 1, 0),
        ]

        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        self.explosion_factor = 0

    def rotate(self, x_angle, y_angle, z_angle):
        self.rotation_x += x_angle
        self.rotation_y += y_angle
        self.rotation_z += z_angle

        self.rotation_x %= 360
        self.rotation_y %= 360
        self.rotation_z %= 360

    def set_explosion(self, factor):
        old_factor = self.explosion_factor
        self.explosion_factor = max(0, min(factor, 2.0))

        if abs(old_factor - self.explosion_factor) > 0.01:
            if old_factor < self.explosion_factor:
                logger.info(f"큐브 분해 중: {self.explosion_factor:.2f}")
            else:
                logger.info(f"큐브 조립 중: {self.explosion_factor:.2f}")

    def draw(self):
        glPushMatrix()

        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        glRotatef(self.rotation_z, 0, 0, 1)

        pieces_with_depth = []

        for piece in self.pieces:
            x, y, z, _ = piece

            explosion_x = x * self.explosion_factor
            explosion_y = y * self.explosion_factor
            explosion_z = z * self.explosion_factor

            final_x = x + explosion_x
            final_y = y + explosion_y
            final_z = z + explosion_z

            depth = final_z

            pieces_with_depth.append((final_x, final_y, final_z, x, y, z, depth))

        pieces_with_depth.sort(key=lambda p: -p[6])

        for final_x, final_y, final_z, x, y, z, _ in pieces_with_depth:
            glPushMatrix()
            glTranslatef(final_x, final_y, final_z)
            self.draw_cube_piece(x, y, z)
            glPopMatrix()

        glPopMatrix()

    def draw_cube_piece(self, x, y, z):
        vertices = [
            [0.45, 0.45, 0.45], [-0.45, 0.45, 0.45], [-0.45, -0.45, 0.45], [0.45, -0.45, 0.45],
            [0.45, 0.45, -0.45], [-0.45, 0.45, -0.45], [-0.45, -0.45, -0.45], [0.45, -0.45, -0.45]
        ]

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 4, 7, 3),
            (1, 5, 6, 2),
            (0, 1, 5, 4),
            (3, 2, 6, 7)
        ]

        face_colors = [
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0)
        ]

        for i, face in enumerate(faces):
            glBegin(GL_QUADS)
            glColor4fv(face_colors[i])
            for vertex in face:
                glVertex3fv(vertices[vertex])
            glEnd()

        glLineWidth(2.5)
        glColor4f(0.4, 0.4, 0.4, 0.8)
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()

class HandTrackingCube:
    def __init__(self):
        logger.info("Initializing HandTrackingCube")
        self.mp_hands = mp.solutions.hands
        self.hands = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.left_hand = None
        self.right_hand = None

        self.left_pinch = False
        self.right_pinch = False

        self.left_pinch_distance = 0
        self.right_pinch_distance = 0
        self.both_hands_pinch_distance = 0

        self.prev_pinch_distance = 0

        self.error_count = 0
        self.max_errors = 5

        self.cap = None

        pygame.init()
        self.screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Hand Tracking Cube")

        self.setup_opengl()

        self.cube = Cube()

        self.auto_rotate = True

        self.running = True

        self.last_time = time.time()
        self.frame_count = 0
        self.log_interval = 10

    def initialize_mediapipe(self):
        try:
            logger.info("Initializing MediaPipe Hands")
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.error_count = 0
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            return False

    def reinitialize_mediapipe(self):
        try:
            logger.warning("Reinitializing MediaPipe due to errors")
            if hasattr(self, 'hands') and self.hands:
                try:
                    self.hands.close()
                except:
                    pass

            return self.initialize_mediapipe()
        except Exception as e:
            logger.error(f"Failed to reinitialize MediaPipe: {e}")
            return False

    def setup_opengl(self):
        glEnable(GL_DEPTH_TEST)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glClearColor(0.0, 0.0, 0.0, 1.0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (800 / 600), 0.1, 50.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -10.0)

    def detect_gestures(self, results):
        self.left_pinch = False
        self.right_pinch = False
        self.left_hand = None
        self.right_hand = None

        if results and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[idx].classification[0].label

                    if handedness == "Left":
                        self.right_hand = hand_landmarks
                    else:
                        self.left_hand = hand_landmarks

        if self.left_hand:
            try:
                thumb_tip = self.left_hand.landmark[4]
                index_tip = self.left_hand.landmark[8]

                self.left_pinch_distance = math.sqrt(
                    (thumb_tip.x - index_tip.x) ** 2 +
                    (thumb_tip.y - index_tip.y) ** 2
                )

                self.left_pinch = self.left_pinch_distance < 0.08

            except Exception as e:
                logger.error(f"Left hand gesture detection error: {e}")
                self.left_pinch = False

        if self.right_hand:
            try:
                thumb_tip = self.right_hand.landmark[4]
                index_tip = self.right_hand.landmark[8]

                self.right_pinch_distance = math.sqrt(
                    (thumb_tip.x - index_tip.x) ** 2 +
                    (thumb_tip.y - index_tip.y) ** 2
                )

                self.right_pinch = self.right_pinch_distance < 0.08

            except Exception as e:
                logger.error(f"Right hand gesture detection error: {e}")
                self.right_pinch = False

        if self.left_pinch and self.right_pinch and self.left_hand and self.right_hand:
            try:
                left_center_x = (self.left_hand.landmark[4].x + self.left_hand.landmark[8].x) / 2
                left_center_y = (self.left_hand.landmark[4].y + self.left_hand.landmark[8].y) / 2

                right_center_x = (self.right_hand.landmark[4].x + self.right_hand.landmark[8].x) / 2
                right_center_y = (self.right_hand.landmark[4].y + self.right_hand.landmark[8].y) / 2

                self.both_hands_pinch_distance = math.sqrt(
                    (left_center_x - right_center_x) ** 2 +
                    (left_center_y - right_center_y) ** 2
                )

                self.both_hands_pinch_distance = 0.7 * self.both_hands_pinch_distance + 0.3 * self.prev_pinch_distance
                self.prev_pinch_distance = self.both_hands_pinch_distance

                explosion_factor = (self.both_hands_pinch_distance - 0.2) * 2.5
                self.cube.set_explosion(explosion_factor)

            except Exception as e:
                logger.error(f"Hand distance calculation error: {e}")
        else:
            self.both_hands_pinch_distance = self.prev_pinch_distance

    def process_frame(self, frame):
        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = self.hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

            self.detect_gestures(results)

            self.display_gesture_info(image)

            return image

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self.error_count += 1
            return frame

    def display_gesture_info(self, image):
        cv2.rectangle(image, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (300, 150), (255, 255, 255), 2)

        cv2.putText(image, "Hand Gesture Detection", (15, 30),
                    cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)

        y_pos = 60
        cv2.putText(image, "Left Hand:", (15, y_pos),
                    cv2.FONT_ITALIC, 0.6, (255, 255, 255), 2)
        y_pos += 20

        if self.left_hand:
            left_status = "Detected"
            left_pinch = "Pinching" if self.left_pinch else "Not pinching"
            cv2.putText(image, left_pinch, (25, y_pos),
                        cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(image, "Not detected", (25, y_pos),
                        cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

        y_pos += 30
        cv2.putText(image, "Right Hand:", (15, y_pos),
                    cv2.FONT_ITALIC, 0.6, (255, 255, 255), 2)
        y_pos += 20

        if self.right_hand:
            right_status = "Detected"
            right_pinch = "Pinching" if self.right_pinch else "Not pinching"
            cv2.putText(image, right_pinch, (25, y_pos),
                        cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(image, "Not detected", (25, y_pos),
                        cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

        width = image.shape[1]
        distance_panel_width = 350

        cv2.rectangle(image, (width - distance_panel_width - 10, 10), (width - 10, 150), (0, 0, 0), -1)
        cv2.rectangle(image, (width - distance_panel_width - 10, 10), (width - 10, 150), (255, 255, 255), 2)

        cv2.putText(image, "Pinch Distance Measurement", (width - distance_panel_width - 5, 30),
                    cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)

        info_y = 60

        if self.left_pinch and self.right_pinch:
            cv2.putText(image, f"Both Pinch Distance: {self.both_hands_pinch_distance:.3f}",
                        (width - distance_panel_width, info_y),
                        cv2.FONT_ITALIC, 0.6, (255, 255, 255), 2)
            info_y += 30

            bar_start_x = width - distance_panel_width + 10
            bar_end_x = width - 20
            bar_width = bar_end_x - bar_start_x

            scaled_distance = min(self.both_hands_pinch_distance, 1.0)
            filled_width = int(bar_width * scaled_distance)

            cv2.rectangle(image, (bar_start_x, info_y), (bar_end_x, info_y + 15), (100, 100, 100), -1)
            if filled_width > 0:
                cv2.rectangle(image, (bar_start_x, info_y), (bar_start_x + filled_width, info_y + 15), (255, 255, 255), -1)

            info_y += 30

            cv2.putText(image, f"Explosion factor: {self.cube.explosion_factor:.2f}",
                        (width - distance_panel_width, info_y),
                        cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
            info_y += 20

        if self.left_pinch:
            cv2.putText(image, f"Left Pinch Distance: {self.left_pinch_distance:.3f}",
                        (width - distance_panel_width, info_y),
                        cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
            info_y += 20

        if self.right_pinch:
            cv2.putText(image, f"Right Pinch Distance: {self.right_pinch_distance:.3f}",
                        (width - distance_panel_width, info_y),
                        cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
        info_y += 20

    def run(self):
        logger.info("Starting HandTrackingCube")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Failed to open webcam")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.initialize_mediapipe():
            logger.error("Failed to initialize MediaPipe")
            self.cap.release()
            return False

        cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hand Tracking', 640, 480)

        clock = pygame.time.Clock()

        while self.running:
            current_time = time.time()
            delta_time = current_time - self.last_time
            self.last_time = current_time

            success, frame = self.cap.read()
            if not success:
                logger.warning("Failed to read frame")
                continue

            frame = cv2.flip(frame, 1)

            self.frame_count += 1
            processed_frame = self.process_frame(frame)

            cv2.imshow('Hand Tracking', processed_frame)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_a:
                        self.auto_rotate = not self.auto_rotate
                        logger.info(f"Auto-rotate: {'On' if self.auto_rotate else 'Off'}")

            if self.auto_rotate:
                self.cube.rotate(10 * delta_time, 20 * delta_time, 0)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glTranslatef(0.0, 0.0, -10.0)

            self.cube.draw()

            pygame.display.flip()

            if cv2.waitKey(5) & 0xFF == 27:
                logger.info("ESC key pressed, exiting")
                break

            clock.tick(120)

        self.cleanup()
        return True

    def cleanup(self):
        logger.info("Cleaning up resources")

        if hasattr(self, 'hands') and self.hands:
            try:
                self.hands.close()
            except:
                pass

        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

        pygame.quit()

if __name__ == "__main__":
    app = HandTrackingCube()
    app.run()