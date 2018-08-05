from queue import Queue
from threading import Thread, Event

import cv2


class ThreadVideoCapture(Thread):
    """
    Use new thread to capture video frames
    call start() to start capturing
    call stop() to stop capturing
    call get_last_frame() to get last frame

    Only one frame will be stored, there is no frame queue.

    idea:
    https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    doc of thread:
    https://docs.python.org/3/library/threading.html
    queue:
    https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
    """

    def __init__(self, capture_interval_in_sec=0, device_index=0, queue_size=250):
        super().__init__()
        self.capture_interval_in_sec = capture_interval_in_sec
        self.event = Event()
        self.video_capture = cv2.VideoCapture(device_index)
        print("Video reading fps : {0}".format(self.video_capture.get(cv2.CAP_PROP_FPS)))
        self.Q = Queue(maxsize=queue_size)

    def run(self):
        while True:
            if self.capture_interval_in_sec > 0:
                self.event.wait(self.capture_interval_in_sec)
            if self.event.is_set():
                break

            if not self.Q.full():
                succeed, frame = self.video_capture.read()
                if succeed:
                    self.Q.put(frame)
                    # print("Queue size: ", self.Q.qsize())

        self.video_capture.release()

    def stop(self):
        self.event.set()

    def get_last_frame(self):
        return self.Q.get()

    def has_more_frames(self):
        return self.Q.qsize() > 0
