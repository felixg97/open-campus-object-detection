import cv2
import textwrap


class ObjectDetectionApp():

    def __init__(self, model=None, api=None, base_path=None, window_name="Object Detection"):
        self.model = model
        self.api = api
        self.base_path = base_path

        self.window_name = window_name

        self.logo = None
        self.logo_HS_Aalen = None
        self.capture = None
        self.results = None

    def start_capturing(self, cam=0):

        # Fullscreen
        window_name = "Object Detection"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.capture = cv2.VideoCapture()
        self.capture.open(cam)

        self.logo = cv2.imread(
            self.base_path + "/assets/chatgpt_logo.png", cv2.IMREAD_UNCHANGED)

        self.logo_HS_Aalen = cv2.imread(
            self.base_path + "/assets/HS_Aalen_logo.png", cv2.IMREAD_UNCHANGED)

        while True:
            # Capture the video frame
            ret, frame = self.capture.read()

            if self.model:
                # Display the results of the current frame
                self.results = self.model(frame)

                # Read label
                df = self.results.pandas().xyxy[0]

                # Change label in model
                for i in df['class']:
                    # Name of label -> model.names[i]

                    label = self.model.names[i]

                    text = "No description."
                    if self.api:
                        text = self.api.get_label_response(label)
                        pass

                    # Plot description
                    for box in self.results.xyxy[0]:
                        if box[5] == i:
                            xB = int(box[2])
                            xA = int(box[0])
                            yB = int(box[3])
                            yA = int(box[1])

                            self._draw_content(
                                frame, text, xA + 5, yB - 5, xB, yB, xB - xA)

                cv2.imshow(self.window_name, self.results.render()[0])
            else:
                cv2.imshow(self.window_name, frame)

            # the 'q' button is set as the quitting button
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.end_capturing()
                break

    def end_capturing(self, ):
        # After the loop release the cap object
        self.capture.release()

        # Destroy all the windows
        cv2.destroyAllWindows()

    def _draw_content(self, img, label, xA, yA, xB, yB, len_text):

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (xA, yA)
        fontScale = 0.5
        font_color = (255, 255, 255)
        text_color_bg = (146, 146, 148, 0.5)
        thickness = 1
        overlay = img.copy()
        alpha = 0.3

        dic = {"ä": "ae", "Ä": "Ae", "ö": "oe",
               "Ö": "Oe", "ü": "ue", "Ü": "ue"}
        for i, j in dic.items():
            label = label.replace(i, j)

        label = textwrap.wrap(label, int(len_text / 9))

        pos = 15 * (len(label) - 1)

        #cv2.rectangle(overlay, (xA, yA - pos - 15),
         #             (xB, yB), text_color_bg, -1)

        # Better performance: without transparent background form, then comment out line 165 addWeighted(...)
        # cv2.rectangle(img, (xA,yA - pos - 15), (xB, yB), text_color_bg, -1)

        for i in range(0, len(label)):
            cv2.putText(img, label[i], (xA, yA - pos), font, fontScale,
                        font_color, thickness, cv2.LINE_AA)

            pos = pos - 15

        # Draw white rectangle with logo
        image_height, image_width = img.shape[:2]

        rectangle_color = (255, 255, 255)

        rectangle_width_ratio = 0.2
        rectangle_height_ratio = 0.05

        rectangle_width = int(image_width * rectangle_width_ratio)
        rectangle_height = int(image_height * rectangle_height_ratio)

        top_left_corner = (0, image_height - rectangle_height)
        bottom_right_corner = (rectangle_width, image_height)

        cv2.rectangle(img, top_left_corner,
                      bottom_right_corner, rectangle_color, thickness=-1)

        # Draw white rectangle with logo
        image_height, image_width = img.shape[:2]

        rectangle_color = (255, 255, 255)

        rectangle_width_ratio = 0.44
        rectangle_height_ratio = 0.1

        rectangle_width = int(image_width * rectangle_width_ratio)
        rectangle_height = int(image_height * rectangle_height_ratio)

        top_left_corner = (0, image_height - rectangle_height)
        bottom_right_corner = (rectangle_width, image_height)

        cv2.rectangle(img, top_left_corner,
                      bottom_right_corner, rectangle_color, thickness=-1)

        # Draw logo
        padding = 5
        logo_height = rectangle_height - 2 * padding
        logo_width = int(self.logo.shape[1] *
                         (logo_height / self.logo.shape[0]))
        resized_logo = cv2.resize(
            self.logo, (logo_width, logo_height), interpolation=cv2.INTER_AREA)

        logo_x = top_left_corner[0] + padding
        logo_y = top_left_corner[1] + padding
        frame = self._overlay_image(img, resized_logo, logo_x, logo_y)

        # Write message in rectangle
        text_line1 = "The texts are generated with"
        text_line2 = "OpenAI's GPT-3.5"  # INFO: this is hardcoded - be careful
        font_scale = 0.5
        font_thickness = 1
        text_color = (0, 0, 0)  # black

        # Write the first line of text
        text_x = top_left_corner[0] + padding + logo_width + padding
        text_y = top_left_corner[1] + padding + int(3 * padding)
        cv2.putText(frame, text_line1, (text_x, text_y), font,
                    font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Write the second line of text
        text_y += int(4 * padding)
        cv2.putText(frame, text_line2, (text_x, text_y), font,
                    font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Logo HS Aalen
        top_left_corner = (0 + 420, image_height - rectangle_height)
        bottom_right_corner = (rectangle_width + 420, image_height)

        cv2.rectangle(img, top_left_corner,
                      bottom_right_corner, rectangle_color, thickness=-1)

        resized_logo_HS_Aalen = cv2.resize(
            self.logo_HS_Aalen, (logo_width, logo_height), interpolation=cv2.INTER_AREA)

        frame = self._overlay_image(img, resized_logo_HS_Aalen, logo_x + 420, logo_y)

        # Write the first line of text
        text_x = top_left_corner[0] + padding + logo_width + padding
        # text_y = top_left_corner[1] + padding + int(3 * padding)
        text_y = 450
        cv2.putText(frame, "HS Aalen", (text_x, text_y), font,
                    font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Write the second line of text
        text_y = 470
        cv2.putText(frame, "Wirtschaftsinformatik", (text_x, text_y), font,
                    font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Following line overlays transparent rectangle
        # over the image
        # self.results = self.model(cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0))

    def _overlay_image(self, background, foreground, x, y):
        foreground_alpha = foreground[:, :, 3] / 255.0
        background_alpha = 1.0 - foreground_alpha

        for c in range(0, 3):
            background[y:y + foreground.shape[0], x:x + foreground.shape[1], c] = (
                foreground_alpha * foreground[:, :, c] +
                background_alpha *
                background[y:y + foreground.shape[0],
                           x:x + foreground.shape[1], c]
            )

        return background
