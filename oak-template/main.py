import depthai as dai
import cv2
from depthai_nodes.node import ParsingNeuralNetwork
from pythonosc import udp_client

#################################################
###################################################
OSC_IP = "127.0.0.1"
OSC_PORT = 9000
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
print(f"Sending OSC data to {OSC_IP}:{OSC_PORT}")
#########################################################################
##############################################################


remoteConnector = dai.RemoteConnection(httpPort=8082)
device = dai.Device()
camera_sensors = device.getConnectedCameraFeatures()
print("Camera sensors: ", camera_sensors)
print("socket 0", camera_sensors[0].socket)


flood_intensity = 1
FLOOD_STEP = 0.1

# The sensor is 4056 x 3040

# REMEMBER TO PASS DEVICE AS INPUT FOR THE PIPELINE
with dai.Pipeline(device) as pipeline:

    # Define source and output
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_out = cam.requestOutput((1920,1080), dai.ImgFrame.Type.BGR888p)
    # cam_out = cam.requestFullResolutionOutput()

    # Mono camera
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoLeftOut = monoLeft.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)
    leftQueue = monoLeftOut.createOutputQueue()



    manip_zoom = pipeline.create(dai.node.ImageManip)

    # manip.initialConfig.setOutputSize(300, 300, dai.ImageManipConfig.ResizeMode.CENTER_CROP)
    manip_zoom.setMaxOutputFrameSize(4000000)
    #manip_zoom.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip_zoom.initialConfig.addCrop(640,360, 640, 360)

    cam_out.link(manip_zoom.inputImage)


    manip_zoom_output = manip_zoom.out.createOutputQueue()

    # manip_nn_adjustment = pipeline.create(dai.node.ImageManip)

    # manip_zoom.out.link(manip_nn_adjustment.inputImage)

    # manip_nn_adjustment.setMaxOutputFrameSize(4000000)
    # manip_nn_adjustment.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    # manip_nn_adjustment.initialConfig.setOutputSize(640, 480)

    camQ = cam_out.createOutputQueue()


    # face detection model
    det_model_description = dai.NNModelDescription("luxonis/yunet:640x360")
    det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        manip_zoom.out, det_model_description
    )

    # Create output queue for detections
    detQ = det_nn.out.createOutputQueue()

    pipeline.start()
    #pipeline.getDefaultDevice().setIrLaserDotProjectorIntensity(1.0)
    pipeline.getDefaultDevice().setIrFloodLightIntensity(flood_intensity)

    # remoteConnector.registerPipeline(pipeline)

    frame_copy = manip_zoom_output.get().getCvFrame()
    h,w = frame_copy.shape[:2]

    print("width: ", w)
    print("height: ", h)

    while pipeline.isRunning():

        # cam_cv_frame = camQ.get().getCvFrame()

        # cam_cv_frame
        # pipeline.getDefaultDevice().setIrFloodLightIntensity(flood_intensity)

        if manip_zoom_output.has():
            frame = manip_zoom_output.get().getCvFrame()

            # # Get detections if available
            # if detQ.has():
            #     detections = detQ.tryGet().detections
            #     for detection in detections:
            #         rect: dai.RotatedRect = detection.rotated_rect
            #         rect = rect.denormalize(w, h)
            #
            #         # Get the 4 corner points, starting from the top left point and going clockwise around the rect points
            #         points = rect.getPoints()
            #
            #         osc_data = []
            #
            #         # Draw each point as a green circle
            #         for point in points:
            #             x = int(point.x)
            #             y = int(point.y)
            #             cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green filled circles
            #
            #
            #             # osc_data.append(int(point.x)/w)
            #             # osc_data.append(int(point.y)/h)
            #             osc_data.append(int(point.x))
            #             osc_data.append(int(point.y))
            #         client.send_message("/face/points", osc_data)
            #
            #         center = (int((points[0].x + points[1].x)/2), int((points[0].y + points[3].y)/2))
            #         client.send_message("/face/center", [center[0], center[1]])
            #
            #
            #
            #         cv2.circle(frame, center, 5, (0, 0, 255), cv2.FILLED)

            cv2.imshow("Manip frame", frame)

        # if camQ.has():
        #     cv2.imshow("Camera frame", camQ.get().getCvFrame())
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord("a"):
            flood_intensity += 0.1
            if flood_intensity > 1:
                flood_intensity = 1
            pipeline.getDefaultDevice().setIrFloodLightIntensity(flood_intensity)
            print(f"Flood intensity: {flood_intensity}")
        elif key == ord("d"):
            flood_intensity -= 0.1
            if flood_intensity < 0:
                flood_intensity = 0
            pipeline.getDefaultDevice().setIrFloodLightIntensity(flood_intensity)
            print(f"Flood intensity: {flood_intensity}")







