import cv2
import numpy as np


def process_video(video_name: str):

    cap = cv2.VideoCapture(f"videos/{video_name}.mp4")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print((width, height))

    while True:
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (grabbed, frame) = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect circles in the image
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, maxRadius=70)
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        cv2.imshow('Resized frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # save on pressing 'y'
            cv2.destroyAllWindows()
            break

    cap.release()

if __name__ == "__main__":
    process_video("video1")
