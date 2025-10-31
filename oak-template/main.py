import depthai as dai
import cv2

remoteConnector = dai.RemoteConnection(httpPort=8082)
device = dai.Device()
camera_sensors = device.getConnectedCameraFeatures()
print("Camera sensors: ", camera_sensors)
print("socket 0", camera_sensors[0].socket)

# The sensor is 4056 x 3040

# REMEMBER TO PASS DEVICE AS INPUT FOR THE PIPELINE
with dai.Pipeline(device) as pipeline:
    # Define source and output
    cam = pipeline.create(dai.node.Camera).build(camera_sensors[0].socket)
    # cam_out = cam.requestOutput((1920,1080), dai.ImgFrame.Type.NV12)
    cam_out = cam.requestFullResolutionOutput()

    manip = pipeline.create(dai.node.ImageManip)

    # manip.initialConfig.setOutputSize(300, 300, dai.ImageManipConfig.ResizeMode.CENTER_CROP)
    manip.setMaxOutputFrameSize(4000000)
    manip.initialConfig.addCrop(1521,1140, 1014, 760)
    cam_out.link(manip.inputImage)

    manipQ = manip.out.createOutputQueue()
    camQ = cam_out.createOutputQueue()

    pipeline.start()
    # remoteConnector.registerPipeline(pipeline)

    while pipeline.isRunning():

        # cam_cv_frame = camQ.get().getCvFrame()

        # cam_cv_frame

        if manipQ.has():
            cv2.imshow("Manip frame", manipQ.get().getCvFrame())
        if camQ.has():
            cv2.imshow("Camera frame", camQ.get().getCvFrame())
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
