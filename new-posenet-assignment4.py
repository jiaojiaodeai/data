#!/usr/bin/python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils
import numpy as np
import argparse
import sys

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)
font = jetson.utils.cudaFont(size=24)

def calculate_angle(pose, id1, id2, id3):
    """Calculate the joint angle at id2 in degrees, using three keypoints."""
    p1 = pose.Keypoints[id1]
    p2 = pose.Keypoints[id2]
    p3 = pose.Keypoints[id3]

    # Vectors point from the middle joint (elbow/knee) to its neighbours.
    v1 = np.array([p1.x - p2.x, p1.y - p2.y], dtype=float)
    v2 = np.array([p3.x - p2.x, p3.y - p2.y], dtype=float)

    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-8:
        return None

    cosine_angle = np.dot(v1, v2) / norm
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return float(np.degrees(angle))


def get_joint_angle(pose, name1, name2, name3):
    """Find the three joints by name (missing joints return None)."""
    id1 = pose.FindKeypoint(name1)
    id2 = pose.FindKeypoint(name2)
    id3 = pose.FindKeypoint(name3)

    if id1 < 0 or id2 < 0 or id3 < 0:
        return None

    return calculate_angle(pose, id1, id2, id3)


def angle_label(angle):
    return "N/A" if angle is None else "{:.1f} deg".format(angle)


# process frames until the user exits
while True:
    # capture the next image
    img = input.Capture()
    if img is None:  # capture timeout / end of stream
        if not input.IsStreaming():
            break
        continue

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=opt.overlay)

    # calculate and draw both angles for each detected person
    for person_number, pose in enumerate(poses, start=1):
        # Left arm: left_shoulder -> left_elbow -> left_wrist
        left_arm_angle = get_joint_angle(
            pose, 'left_shoulder', 'left_elbow', 'left_wrist'
        )

        # Right leg: right_hip -> right_knee -> right_ankle
        right_leg_angle = get_joint_angle(
            pose, 'right_hip', 'right_knee', 'right_ankle'
        )

        # Put angle values directly on the camera image.
        y = 10 + (person_number - 1) * 68
        font.OverlayText(
            img,
            text="Person {} Left arm: {}".format(
                person_number, angle_label(left_arm_angle)
            ),
            x=10, y=y,
            color=font.White, background=font.Gray40
        )
        font.OverlayText(
            img,
            text="Person {} Right leg: {}".format(
                person_number, angle_label(right_leg_angle)
            ),
            x=10, y=y + 30,
            color=font.White, background=font.Gray40
        )

        print("Person {} | left arm: {} | right leg: {}".format(
            person_number, angle_label(left_arm_angle), angle_label(right_leg_angle)
        ))

    if not poses:
        font.OverlayText(
            img, text="No person detected", x=10, y=10,
            color=font.White, background=font.Gray40
        )

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(
        opt.network, net.GetNetworkFPS()
    ))

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
