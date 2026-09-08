#!/usr/bin/env python3
# ROS node for LiDAR and camera fusion with YOLO-based model for traffic light detection

# Imports
import rospy
import cv2
import torch
import math
import time
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32MultiArray, String, Bool
from jsk_recognition_msgs.msg import BoundingBoxArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped


class ImageLidarYOLOProcessor:
    def __init__(self):
        rospy.init_node("lidar_camera_fusion", anonymous=True)
        rospy.loginfo("Started LiDAR Camera Fusion node...")

        # Subscribers for image, LiDAR clusters, and bounding boxes from JSK
        rospy.Subscriber("/image_raw", Image, self.image_callback)
        rospy.Subscriber("/obstacle_detector/cloud_clusters", PointCloud2, self.lidar_callback)
        rospy.Subscriber('/obstacle_detector/jsk_bboxes', BoundingBoxArray, self.bbox_callback)
        rospy.Subscriber("/ndt_pose", PoseStamped, self.callback_ndt_pose)

        # Publishers
        self.image_proj_pub = rospy.Publisher("/lidar_projection/image", Image, queue_size=10)
        self.classified_image_pub = rospy.Publisher("/classified_image/image", Image, queue_size=10)
        self.classified_lidar_pub = rospy.Publisher("/classified_lidar/pointcloud", PointCloud2, queue_size=10)
        self.yolo_marker_pub = rospy.Publisher("/classified_lidar/labels", MarkerArray, queue_size=10)
        self.fusion_data_pub = rospy.Publisher("/lcf_tl_distance", String, queue_size=10)
        self.decision_pub = rospy.Publisher("/lcf_tl_decision", Bool, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

        # Global Variables
        self.bridge = CvBridge()
        self.image = None
        self.lidar_points = None
        self.lidar_distance = None
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.distances = []
        self.marker_id_map = {}
        self.cx = 0.0
        self.cy = 0.0
        self.cz = 0.0
        self.last_alert_time = None
        self.state = False
        self.timeout_sec = 60.0
        self.trigger_dist = 10.5

        # Load YOLO Model
        self.model = YOLO('/home/tihan/LiDAR_Camera/weights/traffic_lights.pt')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        rospy.loginfo(f"[YOLO] Model loaded on: {self.device}")

        # Camera Intrinsic Matrix (K)
        self.K = np.array([[1168.94605108581, 0, 927.807483744860],
                           [0, 1168.84406455632, 537.591706843338],
                           [0, 0, 1]])

        # Extrinsic Parameters (Roll, Pitch, Yaw converted to radians)
        roll, pitch, yaw = np.deg2rad([91.497877, 0.6923781, 89.176863])
        self.T = np.array([[-0.247279673182098],
                           [0.753715170841914],
                           [-1.59535956144533]])

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        self.R = Rx @ Ry @ Rz
        self.extrinsic_matrix = np.hstack((self.R, self.T))

    def timer_callback(self, event): 
        # Regularly publish decision state
        self.decision_pub.publish(Bool(data=self.state))

    def image_callback(self, msg):
        # Convert incoming image and trigger processing
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if hasattr(self, "extrinsic_matrix"):
                self.process_lidar_and_yolo()
            else:
                rospy.logwarn("Extrinsic matrix not initialized yet.")
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")

    def lidar_callback(self, msg):
        # Convert LiDAR point cloud to numpy array
        self.lidar_points = np.array([[p[0], p[1], p[2]] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)])

    def bbox_callback(self, msg):
        # Store JSK bounding box distances
        self.jsk_boxes = msg.boxes
        self.distances = []
        for i, box in enumerate(msg.boxes):
            x = box.pose.position.x
            y = box.pose.position.y
            z = box.pose.position.z
            dist = math.sqrt((self.current_x - x)**2 + (self.current_y - y)**2 + z**2)
            self.distances.append(dist)

    def callback_ndt_pose(self, data):
        self.current_x = data.pose.position.x
        self.current_y = data.pose.position.y
    
    def process_lidar_and_yolo(self):
        # Project LiDAR points onto the image
        if self.image is None or self.lidar_points is None:
            return

        # Convert LiDAR points to camera frame
        lidar_points_hom = np.hstack((self.lidar_points, np.ones((self.lidar_points.shape[0], 1))))
        lidar_camera = (self.extrinsic_matrix @ lidar_points_hom.T).T
        
        # Project to image plane
        lidar_image = self.K @ lidar_camera.T
        lidar_image /= lidar_image[2, :]
        valid_idx = lidar_camera[:, 2] > 0

        # Filter points within image bounds
        proj_x = lidar_image[0][valid_idx].astype(int)
        proj_y = lidar_image[1][valid_idx].astype(int)
        lidar_valid = self.lidar_points[valid_idx]

        bounds_check = (proj_x >= 0) & (proj_x < self.image.shape[1]) & \
                       (proj_y >= 0) & (proj_y < self.image.shape[0])

        proj_x = proj_x[bounds_check]
        proj_y = proj_y[bounds_check]
        lidar_valid = lidar_valid[bounds_check]

        # Draw projections for visualization
        image_with_proj = self.image.copy()
        for x, y in zip(proj_x, proj_y):
            cv2.circle(image_with_proj, (x, y), 3, (0, 0, 255), -1)

        self.image_proj_pub.publish(self.bridge.cv2_to_imgmsg(image_with_proj, "bgr8"))
        # Run YOLO detection and fuse results
        self.run_yolo(self.image, proj_x, proj_y, lidar_valid)

    def run_yolo(self, img, proj_x, proj_y, lidar_points):
        # Fuse YOLO and LiDAR
        results = self.model(img)
        yolo_boxes, class_labels, bbox_centroids = [], [], []
        classified_image = img.copy()
        classified_points = []
        all_pairs = []
        label_distance_map = {}
        detection_data = {}

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.model.names[int(box.cls[0])]

                # Check which projected LiDAR points fall inside the bounding box
                margin = 10 
                polygon = Polygon([(x1 - margin, y1 - margin), (x2 + margin, y1 - margin), (x2 + margin, y2 + margin), (x1 - margin, y2 + margin)])
                inside_idx = [i for i, (x, y) in enumerate(zip(proj_x, proj_y)) if polygon.contains(Point(x, y))]

                if inside_idx:
                    matched_points = lidar_points[inside_idx]                    
                    dists = np.linalg.norm(matched_points, axis=1) # Compute distances of all matched points
                    min_dist = np.min(dists)
                    threshold = 1.05 * min_dist # Adaptive thresholding — take points within 5% of min distance
                    near_indices = np.where(dists <= threshold)[0]
                    # Fallback if too few points (e.g. <3), take top 5 closest
                    if len(near_indices) < 3:
                        near_indices = np.argsort(dists)[:min(5, len(dists))]

                    near_points = matched_points[near_indices]
                    near_dists = dists[near_indices]
                    if self.lidar_distance is not None:
                        lidar_topic_str = f"{self.lidar_distance:.4f}"
                    else:
                        lidar_topic_str = "NaN"

                    print(f"Min: {np.min(near_dists):.4f}, Mean: {np.mean(near_dists):.4f}, "
                          f"Standard Deviation: {np.std(near_dists):.4f}, LiDAR Topic: {lidar_topic_str}")
                    
                    centroid = np.mean(near_points, axis=0) # Compute mean of filtered near points
                    centroid_dist = np.linalg.norm(centroid)
                    min_distance = np.min(near_dists)

                    # Check which projected LiDAR points fall inside the bounding box
                    if self.distances:
                        closest_idx = np.argmin(np.abs(np.array(self.distances) - centroid_dist))
                        self.lidar_distance = self.distances[closest_idx]

                        if closest_idx < len(self.jsk_boxes):
                            box = self.jsk_boxes[closest_idx]
                            self.cx = box.pose.position.x
                            self.cy = box.pose.position.y
                            self.cz = box.pose.position.z

                    # try:
                    #     # Project MIN point
                    #     min_point = near_points[np.argmin(near_dists)]
                    #     min_cam = self.R @ min_point.reshape(3, 1) + self.T
                    #     min_img = self.K @ min_cam
                    #     min_img /= min_img[2]
                    #     min_x, min_y = int(min_img[0]), int(min_img[1])
                    #     cv2.circle(classified_image, (min_x, min_y), 3, (0, 0, 255), -1)  # Red
                    #     cv2.putText(classified_image, "MIN", (min_x + 5, min_y - 5), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 3)

                    #     # Project MEAN (centroid) point
                    #     centroid_cam = self.R @ centroid.reshape(3, 1) + self.T
                    #     centroid_img = self.K @ centroid_cam
                    #     centroid_img /= centroid_img[2]
                    #     cx, cy = int(centroid_img[0]), int(centroid_img[1])
                    #     cv2.circle(classified_image, (cx, cy), 3, (255, 255, 0), -1)  # Cyan
                    #     cv2.putText(classified_image, "MEAN", (cx + 5, cy - 5), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 0), 3)

                    #     if hasattr(self, 'cx') and hasattr(self, 'cy') and hasattr(self, 'cz'):
                    #         lidar_point = np.array([[self.cx], [self.cy], [self.cz]])
                    #         lidar_cam = self.R @ lidar_point + self.T
                    #         lidar_img = self.K @ lidar_cam
                    #         lidar_img /= lidar_img[2]
                    #         lx, ly = int(lidar_img[0]), int(lidar_img[1])
                    #         cv2.circle(classified_image, (lx, ly), 3, (0, 255, 255), -1)  # Yellow
                    #         cv2.putText(classified_image, "LIDAR", (lx + 5, ly - 5), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 3)

                    # except Exception as e:
                    #     rospy.logwarn(f"Projection failed for MIN/MEAN/LiDAR point: {e}")
                    
                    # Store for priority selection later (one per label)
                    if self.lidar_distance is not None:
                        diff = abs(centroid_dist - self.lidar_distance) if self.lidar_distance else 0.0
                        legend_lines = [(f"Label: {label}", (0, 255, 0)),
                                        (f"Mean: {centroid_dist:.4f} m", (255, 255, 0)),       
                                        (f"Nearest: {min_distance:.4f} m", (0, 0, 255)),
                                        (f"LiDAR Distance: {self.lidar_distance:.4f} m" if self.lidar_distance else "LiDAR Distance: NaN", (0, 255, 255)),
                                        (f"Difference: {diff:.4f} m", (255, 255, 255))]
                        entry = {
                            "distance": min_distance,
                            "legend": legend_lines,
                            "image": classified_image.copy() } 

                        if label not in detection_data:
                            detection_data[label] = []
                        detection_data[label].append(entry)
                    
                    # Accumulate data for publishing
                    bbox_centroids.append(centroid)
                    classified_points.extend(near_points)
                    yolo_boxes.append((x1, y1, x2, y2))
                    class_labels.append(label)
                    
                    # For each detected label, if there are multiple entries, take the nearest one 
                    if self.lidar_distance is not None:
                        if label not in label_distance_map:
                            label_distance_map[label] = self.lidar_distance
                        else:
                            label_distance_map[label] = min(label_distance_map[label], self.lidar_distance)

                    # Visualize best entry (closest one) per label
                    for label, entries in detection_data.items():
                        best_entry = min(entries, key=lambda x: x["distance"])
                        legend_lines = best_entry["legend"]
                        image_to_draw = best_entry["image"]

                        font = cv2.FONT_HERSHEY_DUPLEX
                        font_scale = 2
                        thickness = 3
                        line_height = 70
                        padding = 10

                        box_width = max(cv2.getTextSize(text, font, font_scale, thickness)[0][0] for text, _ in legend_lines)
                        box_height = line_height * len(legend_lines)

                        image_h, image_w, _ = image_to_draw.shape
                        x_start = image_w - box_width - 2 * padding
                        y_start = image_h - box_height - 2 * padding

                        cv2.rectangle(image_to_draw, (x_start, y_start), (image_w, image_h), (0, 0, 0), -1)

                        for i, (line, color) in enumerate(legend_lines):
                            y = y_start + padding + (i + 1) * line_height - 5
                            cv2.putText(image_to_draw, line, (x_start + padding, y), font, font_scale, color, thickness)
                        classified_image = image_to_draw

                    cv2.rectangle(classified_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 2
                    thickness = 3
                    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
                    text_w, text_h = text_size
                    text_x = x1
                    text_y = y1 - 10
                    cv2.rectangle(classified_image, (text_x, text_y - text_h), (text_x + text_w, text_y + 4), (0, 0, 0), -1)
                    cv2.putText(classified_image, label, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

                    # Decision Making based on traffic light conditions
                    now = time.time()
                    red_or_yellow_close = False
                    for lbl, d in label_distance_map.items():
                        if lbl.lower() in ['red'] and d <= self.trigger_dist:
                            red_or_yellow_close = True
                            self.last_alert_time = now
                            if not self.state:
                                self.state = True
                                rospy.loginfo(f"[LC Fusion] Red/Yellow detected at {d:.4f} m — sending TRUE")
                                self.decision_pub.publish(Bool(data=True))
                            break

                    # Turn off on green
                    if 'green' in label_distance_map and self.state:
                        self.state = False
                        self.last_alert_time = None
                        rospy.loginfo("[LC Fusion] Green detected — sending FALSE")
                        self.decision_pub.publish(Bool(data=False))

                    # Timeout reset if nothing detected for 30 seconds
                    if self.state and not red_or_yellow_close:
                        if self.last_alert_time and (now - self.last_alert_time > self.timeout_sec):
                            self.state = False
                            self.last_alert_time = None
                            rospy.loginfo("[LC Fusion] Timeout — sending FALSE")
                            self.decision_pub.publish(Bool(data=False))


        # Publish final classified image
        self.classified_image_pub.publish(self.bridge.cv2_to_imgmsg(classified_image, "bgr8"))
        self.publish_classified_data(classified_points, class_labels, bbox_centroids)
        
        # Publish "label: distance" message
        if label_distance_map:
            combined_msg = ", ".join([f"{label}: {dist:.4f}" for label, dist in label_distance_map.items()])
            self.fusion_data_pub.publish(combined_msg)
        else:
            self.fusion_data_pub.publish("unknown: NaN")

    def publish_classified_data(self, points, class_labels, bbox_centroids):
         # Publish point cloud and markers
        if not points:
            self.clear_all_markers()
            return

        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "livox_frame"
        pc_msg = pc2.create_cloud_xyz32(header, np.array(points))
        self.classified_lidar_pub.publish(pc_msg)

        if not hasattr(self, 'marker_id_map'):
            self.marker_id_map = {}

        marker_array = MarkerArray()
        active_ids = set()

        for i, (centroid, label) in enumerate(zip(bbox_centroids, class_labels)):
            if np.isnan(centroid).any():
                continue
            if label not in self.marker_id_map:
                self.marker_id_map[label] = len(self.marker_id_map)
            marker_id = self.marker_id_map[label]
            active_ids.add(marker_id)

            marker = Marker()
            marker.header.frame_id = "livox_frame"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "yolo_classes"
            marker.id = marker_id
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.position.x = centroid[0]
            marker.pose.position.y = centroid[1]
            marker.pose.position.z = centroid[2] + 0.5 + (i * 0.1)
            marker.scale.z = 0.5
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.text = label
            marker_array.markers.append(marker)

        self.delete_old_markers(active_ids)
        self.yolo_marker_pub.publish(marker_array)

    def delete_old_markers(self, active_ids):
        # Remove markers not used in current frame
        marker_array = MarkerArray()
        for marker_id in list(self.marker_id_map.values()):
            if marker_id not in active_ids:
                marker = Marker()
                marker.header.frame_id = "livox_frame"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "yolo_classes"
                marker.id = marker_id
                marker.action = Marker.DELETE
                marker_array.markers.append(marker)
        if marker_array.markers:
            self.yolo_marker_pub.publish(marker_array)

    def clear_all_markers(self):
        # Clear all markers and point cloud when no detection
        marker_array = MarkerArray()
        for marker_id in self.marker_id_map.values():
            marker = Marker()
            marker.header.frame_id = "livox_frame"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "yolo_classes"
            marker.id = marker_id
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        self.yolo_marker_pub.publish(marker_array)

        empty_header = rospy.Header()
        empty_header.stamp = rospy.Time.now()
        empty_header.frame_id = "livox_frame"
        empty_cloud = pc2.create_cloud_xyz32(empty_header, [])
        self.classified_lidar_pub.publish(empty_cloud)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    processor = ImageLidarYOLOProcessor()
    processor.run()