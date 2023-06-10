import numpy as np
import cv2 as cv
import os

def optical_flow(frames):
    frames = np.swapaxes(np.swapaxes(frames, 1, 2), 2, 3).astype(np.uint8)
    res    = np.zeros((frames.shape[0] - 1, frames.shape[1], frames.shape[2], 2))
    prvs = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)
    hsv  = np.zeros_like(frames[0])
    hsv[..., 1] = 255
    for i in range(1, len(frames)):
        frame = frames[i]
        next = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        res[i-1] = np.copy(flow)
        if (cv.waitKey(30) & 0xff) == 27: break
        prvs = next
    cv.destroyAllWindows()
    res = np.swapaxes(np.swapaxes(res, 2, 3), 1, 2)
    return res

def optical_flow_pair(frame_1, frame_2):
    frames = np.stack((frame_1, frame_2), axis = 0)
    assert len(frames.shape) == 4
    assert frames.shape[0] == 2
    assert frames.shape[1] == 3
    frames = np.swapaxes(np.swapaxes(frames, 1, 2), 2, 3).astype(np.uint8)
    assert frames.shape[0] == 2
    assert frames.shape[-1] == 3
    assert len(frames.shape) == 4
    res    = np.zeros((frames.shape[1], frames.shape[2], frames.shape[3] - 1))
    prvs = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)
    hsv  = np.zeros_like(frames[0])
    hsv[..., 1] = 255
    for i in range(1, len(frames)):
        frame = frames[i]
        next = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        res = np.copy(flow)
        if (cv.waitKey(30) & 0xff) == 27: break
        prvs = next
    cv.destroyAllWindows()
    res = np.swapaxes(np.swapaxes(res, 1, 2), 0, 1)
    return res