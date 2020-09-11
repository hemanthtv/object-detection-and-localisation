import cv2
import numpy as np

cap=cv2.VideoCapture(0)
wht=320
confThreshold= 0.5
nmsThreshold=0.3
classesFiles='coco.names'
classNames=[]
with open(classesFiles,'rt') as f:
        classNames= f.read().rstrip('\n').split()

modelConfiguaration="yolov3.cfg"
modelWeights="yolov3.weights"

net=cv2.dnn.readNetFromDarknet(modelConfiguaration,modelWeights)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs,frames):
    hT,wT,cT=frames.shape
    bbox=[]
    classIds=[]
    confs=[]

    for output in outputs:
        for det in output:
            scores=det[5:]
            classId=np.argmax(scores)
            confidence=scores[classId]
            if confidence > confThreshold:
                w,h=int(det[2]*wT) , int(det[3]*hT)
                x,y=int((det[0]*wT) - wT/2),int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices=cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    for i in indices:
        i=i[0]
        box=bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(frames,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(frames,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
        (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)



while True:
    ref,frames=cap.read()

    blob=cv2.dnn.blobFromImage(frames,1/255,(wht,wht),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames=net.getLayerNames()
    outputNames=[layerNames[i[0]-1]for i in net.getUnconnectedOutLayers()]
    #print(outputNames )
    #print(net.getUnconnectedOutLayers())

    outputs=net.forward(outputNames)



    findObjects(outputs,frames)

    cv2.imshow("Images",frames)
    cv2.waitKey(1)