# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs hand tracking and object detection on camera frames using OpenCV. 2 EDGETPU
"""
import argparse
import collections
import common
import cv2
import numpy as np
import os
import math
from PIL import Image
import re
from edgetpu.detection.engine import DetectionEngine

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk


import time
import svgwrite
import gstreamer
from pose_engine import PoseEngine
import tflite_runtime.interpreter as tflite

#==============================
EDGES = (
    ('nose', 'left eye'),
    ('nose', 'right eye'),
    ('nose', 'left ear'),
    ('nose', 'right ear'),
    ('left ear', 'left eye'),
    ('right ear', 'right eye'),
    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle'),
)

def shadow_text(cv2_im, x, y, text, font_size=16):
    cv2_im = cv2.putText(cv2_im, text, (x + 1, y + 1),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    #dwg.add(dwg.text(text, insert=, fill='black',
    #                 font_size=font_size, style='font-family:sans-serif'))
    #dwg.add(dwg.text(text, insert=(x, y), fill='white',
    #                 font_size=font_size, style='font-family:sans-serif'))

def draw_pose(cv2_im, cv2_sodidi, pose, numobject, src_size, color='yellow', threshold=0.2):
    box_x = 0
    box_y = 0  
    box_w = 641
    box_h = 480
    scale_x, scale_y = src_size[0] / box_w, src_size[1] / box_h
    xys = {}
    for label, keypoint in pose.keypoints.items():        
        if keypoint.score < threshold: continue
        # Offset and scale to source coordinate space.
        kp_y = int((keypoint.yx[0] - box_y) * scale_y)
        kp_x = int((keypoint.yx[1] - box_x) * scale_x)
        cv2_im = cv2.putText(cv2_im, str(numobject),(kp_x + 1, kp_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        xys[label] = (numobject,kp_x, kp_y)
        
        cv2.circle(cv2_im,(int(kp_x),int(kp_y)),5,(0,255,255),-1)

    return xys

def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)

def main():
    default_model_dir = '../all_models'
    default_model = 'posenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--top_k', type=int, default=1,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=str, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='classifier score threshold')
    args = parser.parse_args()

    #print('Loading Handtracking model {} with {} labels.'.format(args.model, args.labels))

    #engine = DetectionEngine(args.model)
    #labels = load_labels(args.labels)
    #=====================================================================
    src_size = (640, 480)
    print('Loading Pose model {}'.format(args.model))
    engine = PoseEngine(args.model)

    cap = cv2.VideoCapture(args.camera_idx)
    
    while cap.isOpened():
    
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame
        
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        #declare new window for show pose in 2d plane========================
        h_cap, w_cap, _ = cv2_im.shape
        cv2_sodidi = np.zeros((h_cap,w_cap,3), np.uint8)
        #======================================pose processing=================================
        poses, inference_time = engine.DetectPosesInImage(np.uint8(pil_im.resize((641, 481), Image.NEAREST)))
        #print('Posese is',poses)
        n = 0
        sum_process_time = 0
        sum_inference_time = 0
        ctr = 0
        fps_counter  = avg_fps_counter(30)
        
        input_shape = engine.get_input_tensor_shape()

        inference_size = (input_shape[2], input_shape[1])


        #print('Shape is',input_shape)
        #print('inference size is:',inference_size)
        start_time = time.monotonic()
        
        end_time = time.monotonic()
        n += 1
        sum_process_time += 1000 * (end_time - start_time)
        sum_inference_time += inference_time

        avg_inference_time = sum_inference_time / n
        text_line = 'PoseNet: %.1fms (%.2f fps) TrueFPS: %.2f' % (
            avg_inference_time, 1000 / avg_inference_time, next(fps_counter)
        )
        
        shadow_text(cv2_im, 10, 20, text_line)
        numobject = 0
        xys={}
        #draw_pose(cv2_im, poses, dis, src_size)
        for pose in poses:
            xys = draw_pose(cv2_im,cv2_sodidi, pose, numobject, src_size)
            
        for a, b in EDGES:
            if a not in xys or b not in xys: continue
            num,ax, ay = xys[a]
            num,bx, by = xys[b]
            cv2.line(cv2_im,(ax, ay), (bx, by),(0,0,255))
        
        cv2.imshow('frame', cv2_im)
        cv2.imshow('1', cv2_sodidi)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
