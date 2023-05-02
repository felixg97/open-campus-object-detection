import cv2

from PIL import Image


class WindowCapture():

    def __init__(self, model=None, api=None, window_name="Object Detection"):
        self.model = model
        self.api = api

        self.window_name = window_name

        self.detection_active = False
        self.capture = None

    def toggle_detection(self):
        if not self.model:
            raise Exception("No model loaded!")
        self.detection_active = not self.detection_active

    def start_capturing(self, cam=0):
        # capture.open(0)  # -> internal Webcam
        # capture.open(1)  # -> external Webcam
        self.capture = cv2.VideoCapture()
        self.capture.open(cam)

        while True:
            # Capture the video frame
            ret, frame = self.capture.read()

            if self.model and self.detection_active:
                # Display the resulting frame
                results = self.model(frame)

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

                self._plot_description(frame, "text...", xA+5, yB-5)

                cv2.imshow("Object Detection", results.render()[0])
            else:
                cv2.imshow("Object Detection", frame)

            # the 'q' button is set as the quitting button
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.end_capturing()
                break

    def end_capturing(self,):
        # After the loop release the cap object
        self.capture.release()

        # Destroy all the windows
        cv2.destroyAllWindows()

    def _plot_description(self, img, label, x, y):

        # retrieve description from api
        text = "No description"
        if self.api:
            self.api.get_dummy_response()

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
