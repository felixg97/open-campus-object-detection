#pip install opencv-python
import cv2
import torch
import textwrap


def plot_description(img, label, x, y, len_text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    font_color = (255, 255, 255)
    thickness = 1

    dic = {"ä":"ae", "Ä":"Ae", "ö":"oe", "Ö":"Oe", "ü":"ue", "Ü":"ue"}
    for i, j in dic.items():
        label = label.replace(i, j)

    label = textwrap.wrap(label, int(len_text / 9))
    pos = 15 * (len(label)-1)

    for i in range(0, len(label)):
        cv2.putText(img, label[i], (x, y-pos), font, fontScale,
                    font_color, thickness, cv2.LINE_AA)
        pos = pos - 15

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

        # Read label from detected objects
        df = results.pandas().xyxy[0]

        # Change label in model
        for i in df['class']:

            # Name of label -> model.names[i]

            # Plot description
            for box in results.xyxy[0]:
                if box[5] == i:
                    xB = int(box[2])
                    xA = int(box[0])
                    yB = int(box[3])
                    yA = int(box[1])

                    plot_description(frame, "Der Mensch ist nach der biologischen Systematik eine Art der Gattung Homo aus der Familie der Menschenaffen", xA+5, yB-5, xB-xA)

        cv2.imshow("Object Detection", results.render()[0])

        # the 'q' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    capture.release()

    # Destroy all the windows
    cv2.destroyAllWindows()
