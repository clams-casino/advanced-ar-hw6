#!/usr/bin/env python3
import numpy as np
import os
import math
import cv2
from renderClass import Renderer

import rospy
import yaml
import sys
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

import rospkg

from dt_apriltags import Detector


TAG_SIZE = 0.065


class ARNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ARNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        self.veh = rospy.get_namespace().strip("/")

        rospack = rospkg.RosPack()
        # Initialize an instance of Renderer giving the model in input.
        self.renderer = Renderer(rospack.get_path(
            'augmented_reality_apriltag') + '/src/models/duckie.obj')

        # bridge between opencv and ros
        self.bridge = CvBridge()

        # construct subscriber for images
        self.camera_sub = rospy.Subscriber(
            'camera_node/image/compressed', CompressedImage, self.callback)
        # construct publisher for AR images
        self.pub = rospy.Publisher(
            '~augemented_image/compressed', CompressedImage, queue_size=10)

        # april-tag detector
        self.at_detector = Detector(searchpath=['apriltags'],
                                    families='tag36h11',
                                    nthreads=4,
                                    quad_decimate=2.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)

        # get camera calibration parameters (homography, camera matrix, distortion parameters)
        self.intrinsics_file = '/data/config/calibrations/camera_intrinsic/' + \
            rospy.get_namespace().strip("/") + ".yaml"
        rospy.loginfo('Reading camera intrinsics from {}'.format(
            self.intrinsics_file))
        intrinsics = self.readYamlFile(self.intrinsics_file)

        self.h = intrinsics['image_height']
        self.w = intrinsics['image_width']
        self.camera_mat = np.array(
            intrinsics['camera_matrix']['data']).reshape(3, 3)

        # Precompute some matricies
        self.camera_params = (
            self.camera_mat[0, 0], self.camera_mat[1, 1], self.camera_mat[0, 2], self.camera_mat[1, 2])
        self.inv_camera_mat = np.linalg.inv(self.camera_mat)
        self.mat_3Dto2D = np.concatenate(
            (np.identity(3), np.zeros((3, 1))), axis=1)
        self.T = np.zeros((4, 4))
        self.T[-1, -1] = 1.0


    def callback(self, data):
        img = self.readImage(data)
        tags = self.at_detector.detect(cv2.cvtColor(
            img, cv2.COLOR_BGR2GRAY), estimate_tag_pose=True, camera_params=self.camera_params, tag_size=TAG_SIZE)

        for tag in tags:
            H_tag2img = tag.homography
            projection_mat = self.projection_matrix(H_tag2img)
            self.renderer.render(img, projection_mat)

        msg = self.bridge.cv2_to_compressed_imgmsg(img, dst_format='jpeg')
        self.pub.publish(msg)


    def projection_matrix(self, homography):
        """
            Write here the compuatation for the projection matrix, namely the matrix
            that maps the camera reference frame to the AprilTag reference frame.
        """
        T_plane = self.inv_camera_mat @ homography
        T_plane = T_plane / np.linalg.norm(T_plane[:, 0]) # estimate scale using rotation matrix basis constraint

        r1 = T_plane[:, 0]
        r2 = T_plane[:, 1]
        t = T_plane[:, 2]

        # Make sure r1 and r2 form an orthogonal basis then generate r3
        r2 = (r2 - np.dot(r2, r1)*r1)
        r2 = r2 / np.linalg.norm(r2)
        r3 = np.cross(r1, r2)

        self.T[:3, :] = np.column_stack((r1, r2, r3, t))

        return self.camera_mat @ self.mat_3Dto2D @ self.T


    def readImage(self, msg_image):
        """
            Convert images to OpenCV images
            Args:
                msg_image (:obj:`CompressedImage`) the image from the camera node
            Returns:
                OpenCV image
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg_image)
            return cv_image
        except CvBridgeError as e:
            self.log(e)
            return []


    def readYamlFile(self, fname):
        """
            Reads the 'fname' yaml file and returns a dictionary with its input.

            You will find the calibration files you need in:
            `/data/config/calibrations/`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         % (fname, exc), type='fatal')
                rospy.signal_shutdown()
                return


    def onShutdown(self):
        super(ARNode, self).onShutdown()


if __name__ == '__main__':
    # Initialize the node
    camera_node = ARNode(node_name='augmented_reality_apriltag_node')
    # Keep it spinning to keep the node alive
    rospy.spin()
