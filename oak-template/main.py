import depthai as dai
import cv2
from depthai_nodes.node import ParsingNeuralNetwork
import numpy as np

remoteConnector = dai.RemoteConnection(httpPort=8082)
device = dai.Device()
camera_sensors = device.getConnectedCameraFeatures()
print("Camera sensors: ", camera_sensors)
print("socket 0", camera_sensors[0].socket)

# The sensor is 4056 x 3040

# REMEMBER TO PASS DEVICE AS INPUT FOR THE PIPELINE
with dai.Pipeline(device) as pipeline:

    # Define source and output
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    # cam_out = cam.requestOutput((1920,1080), dai.ImgFrame.Type.NV12)
    cam_out = cam.requestFullResolutionOutput()

    manip_zoom = pipeline.create(dai.node.ImageManip)

    # manip.initialConfig.setOutputSize(300, 300, dai.ImageManipConfig.ResizeMode.CENTER_CROP)
    manip_zoom.setMaxOutputFrameSize(4000000)
    manip_zoom.initialConfig.addCrop(1521,1140, 1014, 760)

    cam_out.link(manip_zoom.inputImage)


    manip_zoom_output = manip_zoom.out.createOutputQueue()

    manip_nn_adjustment = pipeline.create(dai.node.ImageManip)

    manip_zoom.out.link(manip_nn_adjustment.inputImage)

    manip_nn_adjustment.setMaxOutputFrameSize(4000000)
    manip_nn_adjustment.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip_nn_adjustment.initialConfig.setOutputSize(640, 480)

    camQ = cam_out.createOutputQueue()


    # face detection model
    det_model_description = dai.NNModelDescription("luxonis/yunet:640x480")
    det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        manip_nn_adjustment.out, det_model_description
    )

    # Create output queue for detections
    detQ = det_nn.out.createOutputQueue()

    pipeline.start()
    # remoteConnector.registerPipeline(pipeline)

    frame_copy = manip_zoom_output.get().getCvFrame()
    h,w = frame_copy.shape[:2]

    print("width: ", w)
    print("height: ", h)

    while pipeline.isRunning():

        # cam_cv_frame = camQ.get().getCvFrame()

        # cam_cv_frame

        if manip_zoom_output.has():
            frame = manip_zoom_output.get().getCvFrame()

            # Get detections if available
            if detQ.has():
                detections = detQ.tryGet().detections
                for detection in detections:
                    rect: dai.RotatedRect = detection.rotated_rect
                    rect = rect.denormalize(w, h)

                    # Get the 4 corner points
                    points = rect.getPoints()

                    # Draw each point as a green circle
                    for point in points:
                        x = int(point.x)
                        y = int(point.y)
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green filled circles

            cv2.imshow("Manip frame", frame)

        # if camQ.has():
        #     cv2.imshow("Camera frame", camQ.get().getCvFrame())
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
