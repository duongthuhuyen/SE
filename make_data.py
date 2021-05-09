# import the necessary packages

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import glob


from sklearn.metrics import pairwise_distances

net = cv2.dnn.readNetFromDarknet('yolo//yolov3.cfg',
                                 'yolo//yolov3.weights', )
classes = []

with open('yolo//coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layernames = net.getLayerNames()
outputlayers = [layernames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.9,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
                help="# of skip frames between detections")
ap.add_argument("-r", "--recognizer", type=str,
                help="path to recognition model")

args = vars(ap.parse_args())


#if a video path was not supplied, grab a reference to the webcam
if not args.get('input', False):
    print("[INFO] starting video stream ...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

else:
    vs = cv2.VideoCapture(args["input"])

writer = None
trackers = []

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
W = None
H = None

fps = FPS().start()

i_save = 0

n_old = len(glob.glob('labels/*.txt'))
print('Number of existed data:', n_old)

if n_old > 0:
    i_save = n_old

while True:
    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if args["input"] is not None and frame is None:
        break

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    frame = cv2.resize(frame, (416, 256))
    original_frame = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # if we are supposed to be writing a video to disk, initialize
    # the writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (W, H), True)

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker

    if totalFrames % args["skip_frames"] == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 1 / 255., (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(outputlayers)

        # loop over the detections
        centroids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > args['confidence'] and classes[class_id] == 'person':
                    print()
                    print(confidence)
                    center_x = int(detection[0] * W)
                    center_y = int(detection[1] * H)
                    w = int(detection[2] * W)
                    h = int(detection[3] * H)
                    x = center_x - w // 2
                    y = center_y - h // 2

                    if len(centroids) > 0:
                        d = pairwise_distances(centroids, [[center_x, center_x]])
                        if np.min(d) < 50:
                            continue

                    centroids.append([center_x, center_y])

                    print(x, y, w, h)

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(x, y, x+w, y+h)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    trackers.append(tracker)
                    rects.append((x, y, x+w, y+h))
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    if writer is not None:
        writer.write(frame)

    for ir, (startX, startY, endX, endY) in enumerate(rects):
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break

    totalFrames += 1
    fps.update()
    if totalFrames % args["skip_frames"] == 0:
        if len(rects) > 0:
            cv2.imwrite('images/%d_box.jpg' % i_save, frame)
            cv2.imwrite('images/%d.jpg' % i_save, original_frame)
            with open('labels/%d.txt' % i_save, 'w') as f:
                for (startX, startY, endX, endY) in rects:
                    f.write("%d, %d, %d, %d \n" % (startX, startY, endX, endY))

            i_save += 1

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
    vs.stop()

# otherwise, release the video file pointer
else:
    vs.release()

# close any open windows
cv2.destroyAllWindows()
