#! /usr/bin/env python

import os
from collections import deque

from data_utils import*
from publish_utils import*
from kitti_utils import*

DATA_PATH = "/home/jin/KITTI/RawData/2011_09_26/2011_09_26_drive_0005_sync/"
TRACK_PATH = "/home/jin/KITTI/Tracking/label_02/0000.txt"
CALI_PATH = "/home/jin/KITTI/RawData/2011_09_26/"

def compute_3d_box_cam2(h,w,l,x,y,z,yaw):
    """
    Return :3xn in cam2 coordinate
    """
    R = np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    #
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d_cam2 = np.dot(R,np.vstack([x_corners,y_corners,z_corners]))
    corners_3d_cam2 += np.vstack([x,y,z])
    return corners_3d_cam2

class Object():
    def __init__(self,center):
        self.locations = deque(maxlen = 20)
        # self.locations.appendleft(center)

    def update(self, center, dis, theta):
        if len(self.locations) != 0:
            self.locations = np.c_[self.locations,np.ones(len(self.locations))]
            R = np.array([[np.cos(theta),np.sin(theta),-dis],
                        [-np.sin(theta),np.cos(theta),0]])
            self.locations = deque(np.dot(R,self.locations.T).T,maxlen = 20)
        if center is not None:
            self.locations.appendleft(center)

    def reset(self):
        self.locations = deque(maxlen = 20)

if __name__ == '__main__':
    frame = 0
    rospy.init_node('kitti_node',anonymous = True)
    cam_pub = rospy.Publisher('kitti_cam', Image, queue_size = 10)
    pcl_pub = rospy.Publisher('kitti_point_cloud', PointCloud2,queue_size = 10)
    ego_pub = rospy.Publisher('kitti_ego_car', MarkerArray,queue_size = 10)
    imu_pub = rospy.Publisher('kitti_imu',Imu,queue_size = 10)
    gps_pub = rospy.Publisher('kitti_gps',NavSatFix,queue_size = 10)
    box_3d_pub = rospy.Publisher('kitti_3d_box',MarkerArray,queue_size = 10)
    loc_pub = rospy.Publisher('kitti_loc',MarkerArray,queue_size = 10)
    
    bridge = CvBridge()
    df_tracking = read_tracking(TRACK_PATH)
    calib = Calibration(CALI_PATH, from_video = True)

    tracker = {} # dictionary{track_id : Object()}
    pre_imu_data = None
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():

        boxes_2d = np.array(df_tracking[df_tracking.frame == frame][['bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom']])
        types = np.array(df_tracking[df_tracking.frame == frame]['type'])
        track_ids = np.array(df_tracking[df_tracking.frame == frame]['track id'])

        corners_3d_velos = []
        centers = {} # dictionary{track_id : center(x,y)}
        boxes_3d = np.array(df_tracking[df_tracking.frame == frame][['height','width','length','pos_x', 'pos_y','pos_z','rot_y']])
        for track_id, box_3d in zip(track_ids, boxes_3d):
            corners_3d_cam2 = compute_3d_box_cam2(*box_3d)
            corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
            corners_3d_velos.append(corners_3d_velo)
            centers[track_id] = np.mean(corners_3d_velo, axis = 0)[:2]
        centers[-1] = np.array([0,0])
        img = read_camera(os.path.join(DATA_PATH,'image_02/data/%010d.png'%frame))
        point_cloud = read_point_cloud(os.path.join(DATA_PATH,'velodyne_points/data/%010d.bin'%frame))
        
        imu_data = read_imu(os.path.join(DATA_PATH,'oxts/data/%010d.txt'%frame))
        
        if pre_imu_data is None:
            for track_id in centers:
                tracker[track_id] = Object(centers[track_id])
        else:    
            dis = 0.1 * np.linalg.norm(imu_data[['vf','vl']])
            theta = float(imu_data.yaw - pre_imu_data.yaw)
            for track_id in centers:
                if track_id in tracker:
                    tracker[track_id].update(centers[track_id], dis, theta)
                else:
                    tracker[track_id] = Object(centers[track_id])
            for track_id in tracker:
                if track_id not in centers:
                    # tracker[track_id].reset()
                    tracker[track_id].update(None, dis, theta)

        pre_imu_data = imu_data

        publish_boxedimage(cam_pub, bridge, img, boxes_2d, types)
        publish_boxes_3d(box_3d_pub, corners_3d_velos, types, track_ids)
        publish_point_cloud(pcl_pub, point_cloud)
        publish_ego_car(ego_pub)
        publish_imu_data(imu_pub, imu_data)
        publish_gps_data(gps_pub, imu_data)
        publish_loc(loc_pub, tracker, centers)
        
        rospy.loginfo("published")
        rate.sleep()
        frame += 1
        if frame ==154:
            frame = 0
            for track_id in tracker:
                tracker[track_id].reset()
                  
