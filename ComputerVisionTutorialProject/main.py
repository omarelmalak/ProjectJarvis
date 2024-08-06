import cv2
import mediapipe as mp
import os
import argparse

INPUT_DIRECTORY = "./media/input_media"
OUTPUT_DIRECTORY = "./media/output_media"


class GeneralHelpers:
    @staticmethod
    def configure_args():
        args = argparse.ArgumentParser()
        args.add_argument("--mode", default="video")
        args.add_argument("--filePath", default=f"{INPUT_DIRECTORY}/omar_video.mp4")

        args = args.parse_args()

        return args


class CV2Helpers:
    @staticmethod
    def display_image(image):
        cv2.imshow("Face Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def process_to_rectangle(image, face_detection):
        image_height, image_width, _ = image.shape

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        out = face_detection.process(image_rgb)

        if out.detections is not None:
            for detection in out.detections:
                location_data = detection.location_data
                bbox = location_data.relative_bounding_box

                x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

                x1 = int(x1 * image_width)
                y1 = int(y1 * image_height)
                w = int(w * image_width)
                h = int(h * image_height)

                image[y1: y1 + h, x1:x1 + w, :] = cv2.blur(image[y1:(y1 + h), x1:(x1 + w), :], (30, 30))
                image = cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

        return image


class InputHandlers:
    @staticmethod
    def handle_image(args, face_detection):
        image = cv2.imread(args.filePath)

        image = CV2Helpers.process_to_rectangle(image, face_detection)

        cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, "blurred_radio.png"), image)

    @staticmethod
    def handle_video(args, face_detection):
        video_capture = cv2.VideoCapture(args.filePath)
        ret, frame = video_capture.read()

        output_video = cv2.VideoWriter(os.path.join(OUTPUT_DIRECTORY, "blurred.mp4"),
                                       cv2.VideoWriter_fourcc(*"MP4V"),
                                       25,
                                       (frame.shape[1], frame.shape[0]))

        while ret:
            frame = CV2Helpers.process_to_rectangle(frame, face_detection)
            output_video.write(frame)

            ret, frame = video_capture.read()

        video_capture.release()
        output_video.release()

    @staticmethod
    def handle_webcam(args, face_detection):
        webcam_capture = cv2.VideoCapture(0)

        ret, frame = webcam_capture.read()

        if not webcam_capture.isOpened():
            print("ERROR: Could not open webcam.")
            return

        while ret:
            frame = CV2Helpers.process_to_rectangle(frame, face_detection)

            cv2.imshow("frame", frame)
            cv2.waitKey(25)

        webcam_capture.release()


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    args = GeneralHelpers.configure_args()

    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1) as face_detection:
        if args.mode in ["image"]:
            InputHandlers.handle_image(args, face_detection)
        elif args.mode in ["video"]:
            InputHandlers.handle_video(args, face_detection)
        elif args.mode in ['webcam']:
            InputHandlers.handle_webcam(args, face_detection)
