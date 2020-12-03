import cv2
# import mtcnn
# from threading import Thread, Lock
import queue
# import os
# from PIL import Image
from faceDetect import detectFace
from imutils.video import WebcamVideoStream

# node = Lock()
q = queue.Queue()
# face_detector = mtcnn.MTCNN()
# vc = WebcamVideoStream(src=0).start()
# conf_t = 0.90

def runApp(vc):    
    # while vc.isOpened():
    while True:
        frame = vc.read()
        if frame.shape != None:
            frame = cv2.resize(frame, (300,300))

        # q.put(frame)
        frame = detectFace(frame)
        cv2.imshow("Recognition Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            vc.stop()
            break
        
# def detectFace():
#     node.acquire()
#     frame = q.get()
#     q.task_done()
#     # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_detector.detect_faces(frame)
#     for qres in results:
#         x1, y1, width, height = res['box']
#         x1, y1 = abs(x1), abs(y1)
#         x2, y2 = x1 + width, y1 + height

#         confidence = res['confidence']
#         if confidence < conf_t:
#             continue
#         key_points = res['keypoints'].values()

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
#         cv2.putText(frame, f'conf: {confidence:.3f}', (x1, y1), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)

#         for point in key_points:
#             cv2.circle(frame, point, 5, (0, 255, 0), thickness=-1)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#     node.release()


# def threadApp():
#     processors = []
#     for i in range(os.cpu_count()):
#         processors.append(Thread(target=detectFace, args=()))
#     for process in processors:
#         process.start()
#     for process in processors:
#         process.join()


if __name__ == "__main__":
    vc = WebcamVideoStream(src=0).start()
    runApp(vc)
    cv2.destroyAllWindows()
    # while True:
    #     threadApp()