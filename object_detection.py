
import cv2
import torch

from PIL import Image


def plot_description(img, label, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (x, y)
    fontScale = 0.5
    font_color = (255, 255, 255)
    thickness = 1
    # text_color_bg=(0, 0, 0)

    # x, y = org
    # text_size, _ = cv2.getTextSize(label, font, fontScale, thickness)
    # text_w, text_h = text_size
    # cv2.rectangle(img, org, (x + text_w, y - text_h), color, -1)
    cv2.putText(img, label, org, font, fontScale,
                font_color, thickness, cv2.LINE_AA)

    # Generate Colors
    # names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    # colors=[colors(x, True) for x in 1000[:, 5]]


if __name__ == "__main__":

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    capture = cv2.VideoCapture()
    capture.open(0)  # -> internal Webcam
    # capture.open(1)  # -> external Webcam

    # Fullscreen
    # window_name = "Object Detection"
    # cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty(
    #     window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while (True):

        # Capture the video frame
        ret, frame = capture.read()

        # Display the resulting frame
        results = model(frame)

        # Label von erkannten Objekten auslesen
        df = results.pandas().xyxy[0]

        # Label in Modell ändern
        for i in df['class']:

            # Name of label -> model.names[i]

            # plot description
            for box in results.xyxy[0]:
                if box[5] == i:
                    xB = int(box[2])
                    xA = int(box[0])
                    yB = int(box[3])
                    yA = int(box[1])

                    plot_description(frame, "text...", xA+5, yB-5)

        cv2.imshow("Object Detection", results.render()[0])

        # the 'q' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    capture.release()

    # Destroy all the windows
    cv2.destroyAllWindows()
