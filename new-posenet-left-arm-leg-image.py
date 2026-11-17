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

"""PoseNet image example: calculate and draw LEFT elbow and LEFT knee angles.

Usage on Jetson Nano:
    python3 new-posenet-left-arm-leg-image.py test.jpg result.jpg
"""

import argparse
import math
import sys

import jetson.inference
import jetson.utils


def calculate_angle(pose, id1, id2, id3):
    """Return the angle at the middle keypoint (id2), in degrees."""
    p1 = pose.Keypoints[id1]
    p2 = pose.Keypoints[id2]
    p3 = pose.Keypoints[id3]

    # Vectors pointing from the middle joint to its two neighbours.
    v1x, v1y = p1.x - p2.x, p1.y - p2.y
    v2x, v2y = p3.x - p2.x, p3.y - p2.y

    magnitude = math.hypot(v1x, v1y) * math.hypot(v2x, v2y)
    if magnitude < 1e-8:
        return None

    cosine = (v1x * v2x + v1y * v2y) / magnitude
    cosine = max(-1.0, min(1.0, cosine))  # Avoid acos domain errors.
    return math.degrees(math.acos(cosine))


def get_joint_angle(pose, first, middle, last):
    """Look up actual detected keypoints by name, not by fixed list position."""
    id1 = pose.FindKeypoint(first)
    id2 = pose.FindKeypoint(middle)
    id3 = pose.FindKeypoint(last)

    if id1 < 0 or id2 < 0 or id3 < 0:
        return None

    return calculate_angle(pose, id1, id2, id3)


def format_angle(value):
    return "N/A" if value is None else "{:.1f} deg".format(value)


def main():
    parser = argparse.ArgumentParser(description="Measure left arm and left leg angles in one image")
    parser.add_argument("input_image", help="input JPEG/PNG image")
    parser.add_argument("output_image", help="output image, e.g. result.jpg")
    parser.add_argument("--network", default="resnet18-body", help="PoseNet model name")
    parser.add_argument("--overlay", default="links,keypoints", help="PoseNet overlay options")
    parser.add_argument("--threshold", type=float, default=0.15, help="PoseNet confidence threshold")
    opt = parser.parse_args()

    # Load the same pretrained pose estimation model as in the teacher's script.
    net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)

    img = jetson.utils.loadImage(opt.input_image)
    poses = net.Process(img, overlay=opt.overlay)
    font = jetson.utils.cudaFont(size=22)

    print("Detected {} person(s)".format(len(poses)))

    for person_number, pose in enumerate(poses, 1):
        # LEFT ARM: shoulder -> elbow -> wrist; the elbow is the vertex.
        left_arm = get_joint_angle(pose, "left_shoulder", "left_elbow", "left_wrist")

        # LEFT LEG: hip -> knee -> ankle; the knee is the vertex.
        left_leg = get_joint_angle(pose, "left_hip", "left_knee", "left_ankle")

        print("Person {}: left arm = {}, left leg = {}".format(
            person_number, format_angle(left_arm), format_angle(left_leg)))

        # Draw results on the image so they also appear in the saved JPG/PNG.
        y = 10 + (person_number - 1) * 64
        font.OverlayText(img,
                         text="Person {} left arm: {}".format(person_number, format_angle(left_arm)),
                         x=10, y=y, color=font.White, background=font.Gray40)
        font.OverlayText(img,
                         text="Person {} left leg: {}".format(person_number, format_angle(left_leg)),
                         x=10, y=y + 30, color=font.White, background=font.Gray40)

    if not poses:
        font.OverlayText(img, text="No person detected", x=10, y=10,
                         color=font.White, background=font.Gray40)

    jetson.utils.cudaDeviceSynchronize()
    jetson.utils.saveImage(opt.output_image, img)
    print("Saved result: {}".format(opt.output_image))


if __name__ == "__main__":
    main()
