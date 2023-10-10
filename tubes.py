import cv2
import numpy as np
import math
from copy import deepcopy

import time
from pathlib import Path
import depthai as dai


   

     
            


class Tubes:
    def __init__(self):
        #self.img = cv2.imread(img_path)
        self.img = self.get_image()
        self.img = self.crop_image(self.img, 900, 600, 3400, 2800)
        cv2.imwrite('image.png', self.img)
        
        self.img_height, self.img_width, _ = self.img.shape
        self.pts = None
        #self.warped_ipad = self.add_padding(self.findIPAD(self.img), self.img, 1/2)
        self.warped_ipad = self.findIPAD(self.img)

        cv2.imwrite('ipad.png', self.warped_ipad)

        
        self.warped_height, self.warped_width, _ = self.warped_ipad.shape

        self.tube_centers_warped = []
        self.tube_arr, self.tube_imgs = self.findTube(self.warped_ipad)

        if self.tube_centers_warped is not None and self.pts is not None:
            self.tube_centers_original = [
                (int(pt[0] + self.pts[0][0]), int(pt[1] + self.pts[0][1])) for pt in self.tube_centers_warped
            ]
            print(self.tube_centers_original)
            self.plotTubeCenters(self.tube_centers_original)
        else:
            print("Either self.tube_centers_warped or self.pts is None.")

        self.colors, self.gamecolors = self.findTubeColors(self.tube_imgs)

        print(self.colors)

    def get_image(self):

        img = False
        
        pipeline = dai.Pipeline()

        camRgb = pipeline.create(dai.node.ColorCamera)
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)

        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        camRgb.video.link(xoutRgb.input)

        xin = pipeline.create(dai.node.XLinkIn)
        xin.setStreamName("control")
        xin.out.link(camRgb.inputControl)

        # Properties
        videoEnc = pipeline.create(dai.node.VideoEncoder)
        videoEnc.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
        camRgb.still.link(videoEnc.input)

        # Linking
        xoutStill = pipeline.create(dai.node.XLinkOut)
        xoutStill.setStreamName("still")
        videoEnc.bitstream.link(xoutStill.input)

        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:

            # Output queue will be used to get the rgb frames from the output defined above
            qRgb = device.getOutputQueue(name="rgb", maxSize=30, blocking=False)
            qStill = device.getOutputQueue(name="still", maxSize=30, blocking=True)
            qControl = device.getInputQueue(name="control")

            

            # Make sure the destination path is present before starting to store the examples
            dirName = "rgb_data"
            Path(dirName).mkdir(parents=True, exist_ok=True)

            while not img:
                frame = None
                inRgb = qRgb.tryGet()  # Non-blocking call, will return a new data that has arrived or None otherwise
                if inRgb is not None:
                    frame = inRgb.getCvFrame()
                    # 4k / 4
                    frame = cv2.pyrDown(frame)
                    frame = cv2.pyrDown(frame)

                key = cv2.waitKey(1)

                ctrl_initial = dai.CameraControl()
                ctrl_initial.setManualExposure(5000, 600)  # 10000us (10ms), sensitivity 800
                ctrl_initial.setBrightness(8)  # Brightness level 2
                ctrl_initial.setCaptureStill(True)
                qControl.send(ctrl_initial)

                # Add some delay to allow settings to take effect
                time.sleep(3)  # 2 seconds delay
                            
                    

                print("Sent 'still' event to the camera!")

                if qStill.has():
                    fName = f"{dirName}/{int(time.time() * 1000)}.jpeg"
                    with open(fName, "wb") as f:
                        imgData = qStill.get().getData()
                        f.write(imgData)
                        print('Image saved to', fName)
                        img = True

                        # Convert byte array to OpenCV image
                        nparr = np.frombuffer(imgData, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        return frame

            
    def crop_image(self, img, x_start, y_start, x_end, y_end):
        cropped_img = img[y_start:y_end, x_start:x_end]
        return cropped_img

    def findByColor(self, img):

        # Define the lower and upper bounds for the darker black pixels
        lower_bound = np.array([0, 0, 0])
        # You can adjust this value to include more or fewer dark pixels
        upper_bound = np.array([50, 50, 50])

        # Apply the in-range filter to get the mask
        mask = cv2.inRange(img, lower_bound, upper_bound)
        gauss = cv2.GaussianBlur(mask, (3, 3), 0)
        # Perform morphological operations to remove noise if necessary
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(gauss, cv2.MORPH_CLOSE, kernel)
        return mask

    def findThreshold(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gauss = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh_gaussian = cv2.threshold(
            gauss, 0, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow('gauss', thresh_gaussian)
        # cv2.waitKey(0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(thresh_gaussian, cv2.MORPH_CLOSE, kernel)
        return mask

    def clkwBox(self, box):
        ysort = sorted(box, key=lambda x: (x[1]))
        if ysort[0][0] > ysort[1][0]:
            ysort[0], ysort[1] = ysort[1], ysort[0]
        if ysort[2][0] < ysort[3][0]:
            ysort[2], ysort[3] = ysort[3], ysort[2]
        return np.array(ysort)

    def four_point_transform(self, img, box):
        tl, tr, br, bl = box
        # print(tl)
        widthA = np.sqrt(((br[0] - bl[0]) ** 2)+((br[1]-bl[1])**2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2)+((tr[1]-tl[1])**2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2)+((tr[1]-br[1])**2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2)+((tl[1]-bl[1])**2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1,
                                                  maxHeight-1], [0, maxHeight-1]], dtype='float32')

        M = cv2.getPerspectiveTransform(np.float32(box), dst)
        wraped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        return wraped

    def findTube(self, img, area_threshold=0.6):
        thresh = self.findThreshold(img)
        contours = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow('thresh', thresh)
        # cv2.waitKey(0)
        # plt.imshow(contours)
        # plt.show()
        # print(contours[0][1])
        a = cv2.drawContours(img, contours[0], -1, (0, 255, 0), 3)
        # cv2.imshow('Contours', a)
        # cv2.waitKey(0)
        # print(contours)
        rects = []
        tubes = []
        tubes_img = []

        for cnt in contours[0]:
            p = cv2.arcLength(cnt, True)
            epsilon = 0.01 * p
            poly = cv2.approxPolyDP(cnt, epsilon, True)
            rect = list(cv2.boundingRect(poly))
            rect = np.array(rect)
            if rect[3] > rect[2]:
                rects.append(rect)

        rects = np.array(sorted(rects, key=lambda x: (x[0], x[1])))
        MAX_AREA = max(rects[:, 2]) * max(rects[:, 3])

        for rect in rects:

            if rect[2] * rect[3] > area_threshold * MAX_AREA:

                rect[0] += rect[2]/8
                rect[2] -= 2 * rect[2]/8
                rect[1] += rect[3]/6
                rect[3] -= rect[3]/6

                center = [int((rect[0] + rect[0] + rect[2]) / 2),
                          int((rect[1] + rect[1] + rect[3]) / 2) - 30]

                self.tube_centers_warped.append(center)

                rect = [[rect[0], rect[1]], [rect[0] + rect[2], rect[1]],
                        [rect[0] + rect[2], rect[1] + rect[3]], [rect[0], rect[1] + rect[3]]]
                tubes.append(rect)

                tubes_img.append(self.four_point_transform(img, rect))

        return np.array(tubes), tubes_img

    def rgb_euclid(self, color1, color2):
        diff = np.array(color2) - np.array(color1)
        return math.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)

    def add_padding(self, box, img, padding_ratio=1/10):
        height, width, _ = img.shape
        padding_x = width * padding_ratio * 1.3
        padding_y = height * padding_ratio * 2

        padded_box = []
        for point in box:
            x_offset = np.sign(point[0] - width / 2) * padding_x
            y_offset = np.sign(point[1] - height / 2) * padding_y

            # Modify padding based on the y-coordinate of the point
            if point[1] < height / 2:  # If point is in the upper half
                y_offset *= 1  # Double the padding
            else:  # If point is in the lower half
                y_offset *= 1  # Remove the padding

            if point[0] < width / 2:
                x_offset *= 1.1
            else:
                x_offset *= 1.5

            padded_point = [point[0] - x_offset, point[1] - y_offset]
            padded_box.append(padded_point)

        return np.array(padded_box, dtype=int)

    def plotTubeCenters(self, tube_centers):
        # Create a copy of the original image to draw on
        img_with_centers = self.img.copy()

        # Loop through each tube center
        for center in tube_centers:
            # Draw a circle at the center of the tube
            cv2.circle(img_with_centers, (int(center[0]), int(center[1])),
                       radius=5, color=(0, 255, 0), thickness=-1)

        # Display the image with the tube centers
        # cv2.imshow('Image with Tube Centers', img_with_centers)
        # cv2.waitKey(0)
        cv2.imwrite('tubecenters.png', img_with_centers)

    # def findIPAD(self, img):
    #     box = [[700, 400], [2000, 400], [2000, 1600], [700, 1600]]
    #     # Add padding to the box points

    #     # Apply the four-point transform
    #     self.pts = np.array(box, dtype=np.float32)

    #     warped_ipad = self.four_point_transform(img, box)
    #     print(warped_ipad.shape)

    #     return warped_ipad

    def findIPAD(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define the color range for iPads (example values, you may need to adjust these)
        lower_color = np.array([0, 0, 0])        # Lower HSV threshold for black
        upper_color = np.array([180, 255, 30])   # Upper HSV threshold for black

        gauss = cv2.GaussianBlur(hsv_image, (7, 7), 0)
        
        cv2.imwrite('gauss.png', gauss)

        # Create a binary mask based on the color range
        color_mask = cv2.inRange(gauss, lower_color, upper_color)

        cv2.imwrite('color.png', color_mask)

        # Find contours in the mask
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables to track the largest contour and its bounding rectangle
        largest_contour = None
        largest_area = 0

        for contour in contours:
            # Find the area of the current contour
            area = cv2.contourArea(contour)
            if area > largest_area:
                largest_area = area
                largest_contour = contour

        rect = None
        if largest_contour is not None:
            # Find the bounding rectangle of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            rect = (x, y, w, h)
            # Draw the bounding rectangle on the original image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extract the iPad region using four_point_transform
            ipad_image = self.four_point_transform(img, [[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        return ipad_image 

    def plot_color(self, color):
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0
        endX = startX + (300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX
        return bar

    def findTubeColors(self, tubes_img):
        colors = []
        gamecolors = []
        for t, tubes in enumerate(tubes_img):
            color = []
            height = math.floor(tubes.shape[0]/4 - 1)
            width = math.floor(tubes.shape[1]/4 - 1)
            y_top = tubes.shape[0] - height
            y_bot = tubes.shape[0]
            h_pad = math.floor(width / 3)
            v_pad = math.floor(height / 5)
            box_index = 0

            while y_top > 0:
                color_img = tubes[y_top + v_pad: y_bot - 3 *
                                  v_pad,  2 * h_pad: tubes.shape[1] - 2 * h_pad]

                y_top -= height
                y_bot -= height

                color_img = color_img.reshape(
                    (color_img.shape[0] * color_img.shape[1], 3))

                # Convert to np.float32
                color_img = np.float32(color_img)

                # Define criteria, apply kmeans()
                criteria = (cv2.TERM_CRITERIA_EPS +
                            cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                k = 1
                _, _, centers = cv2.kmeans(
                    color_img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

                boxcolor = centers[0]

                color_bar = self.plot_color(boxcolor)
                box_index += 1

                if len(gamecolors) == 0:
                    gamecolors.append(boxcolor)
                    color.append(1)
                elif all(boxcolor < 70):
                    continue
                else:
                    found = False
                    min = []
                    for index, c in enumerate(gamecolors):
                        if (self.rgb_euclid(boxcolor, c) < 70):
                            min.append([self.rgb_euclid(boxcolor, c), index])
                            found = True

                    if not found:
                        gamecolors.append(boxcolor)
                        color.append(len(gamecolors))
                    else:
                        color.append(min[np.argmin(min, axis=0)[0]][1] + 1)
            box_index = 0
            colors.append(color)

        return colors, gamecolors
