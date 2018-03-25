from struct import *
import stateMessage2_pb2
import cv2 as cv
import numpy as np
from numpy.random import choice
import zmq
from keras.models import load_model
from sklearn import preprocessing

# keras

def predictActions(rawData, model):
    ret = 4
    alt = 4
    try:
        bArr = bytearray(rawData)
        npImage = np.frombuffer(bArr, dtype=np.uint8)
        frame = cv.imdecode(npImage, cv.IMREAD_UNCHANGED)

        bw = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        print('hola')
        cv.imshow('feed', bw)
        arr = np.array(bw)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.,1.))
        arr = min_max_scaler.fit_transform(arr)
        input_arr = arr.reshape(1,arr.shape[0],arr.shape[1],1)
        action = model.predict(input_arr)
        # FD BK RT LT NO
        

        ret = action.argmax(axis=1)[0]
        
        smax = action[0][0]
        sindex = 0
        print('Action: {0:}'.format(action))
        for index,value in enumerate(action[0]):
            if value>smax and value < action[0][ret]:
                smax = value
                sindex = index

        alt = sindex

    except Exception as e:
        print("Exception in predictActions: {0:}".format(e))
    cv.waitKey(2)

    print('Ret: {0:} Alt: {1:}'.format(ret, alt))
    ch = choice([ret,alt],1,p=[0.8,0.2])
    return ch[0]

#openCV
def processFrame(rawData):
    bArr = bytearray(rawData)
    npImage = np.frombuffer(bArr, dtype=np.uint8)
    frame = cv.imdecode(npImage, cv.IMREAD_UNCHANGED)

    greenLower = (29, 86, 6)
    greenUpper = (64, 255, 255)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
    mask = cv.inRange(hsv, greenLower, greenUpper)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    #find contours in the  mask and initialize the current (x,y) center
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts)>0:
        #find largest contour in the mask
        c = max(cnts, key=cv.contourArea)
        x,y,w,h = cv.boundingRect(c)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        #cv.rectangle(mask,(x,y),(x+w,y+h),(0,255,255),2)
        #((x,y), radius) = cv.minEnclosingCircle(c)
        #M = cv.moments(c)
        #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        #if radius > 10:
        #    cv.circle(frame, (int(x),int(y)), int(radius),(0,255,255), 2)

    cv.imshow('feed', frame)
    #cv.imshow('process',mask)
    cv.waitKey(2)

def recv_frame(socket,model,flags=0, protocol = -1):
    try:
        data = socket.recv(flags)
        gameState = stateMessage2_pb2.GameState()
        gameState.ParseFromString(data)
        #print('#---Recieved Data---#')
        #print('Proximity        [0]: {0:8.4f}, [1]: {1:8.4f}, [2]: {2:8.4f}, [3]:{3:8.4f}'.format(gameState.sensor0, gameState.sensor1, gameState.sensor2, gameState.sensor3))
        #print('Accelerometers     X: {0:8.4f},   Y: {1:8.4f},   Z: {2:8.4f}'.format(gameState.velX, gameState.velY, gameState.velZ))
        #print('Gyro               X: {0:8.4f},   Y: {1:8.4f},   Z: {2:8.4f}'.format(gameState.rotX, gameState.rotY, gameState.rotZ))
        #processFrame(gameState.image)
        rep = predictActions(gameState.image, model)
        return rep
    except Exception as e:
        print('Exception: {0}'.format(e))
        #print(e)
        print('Continue')
        return 4

def send_string(socket, message="None" , flags=0, protocol = -1):
    socket.send_string(message)



#set up actions
actions = ["FD","BK","RT","LT","ST"]

print("Trying to load model")
model = load_model("./TrainedSet6.h5")
print('Model loaded')


# create context
context = zmq.Context()

# create REP socket
socket = context.socket(zmq.REP)

#bind socket to port
address = "tcp://127.0.0.1:10000"
socket.bind(address)
print("Server bound to " + address)
print('Waiting for connection')
#actions[randint(0,4)]
while True:
    # Wait for a connection
    rep = recv_frame(socket, model)
    print('processed a frame! send reply')
    print('sending {0:}'.format(actions[rep]))
    send_string(socket, actions[rep])
    

