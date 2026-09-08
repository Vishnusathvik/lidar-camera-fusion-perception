# LiDAR–Camera Fusion for Depth-Enhanced Traffic Light Detection

A ROS-based LiDAR–Camera fusion system for detecting traffic lights using a YOLO-based vision model and estimating their distance using 3D LiDAR point-cloud information.

The system combines semantic information from the camera with spatial and depth information from LiDAR to provide distance-aware traffic-light perception for autonomous vehicle navigation.

## Overview

Camera-based object detection provides the semantic class of an object but does not directly provide accurate 3D distance information.

This project addresses this by projecting LiDAR points onto the camera image and associating the projected 3D points with YOLO detection bounding boxes.

The fusion system provides:

- Traffic-light detection and classification
- 3D distance estimation
- LiDAR points associated with detected objects
- Camera–LiDAR projection visualization
- Classified LiDAR point-cloud generation
- Traffic-light based decision making
- RViz visualization for detected objects

## System Pipeline

Camera Image
    |
    v
YOLO Detection
    |
    v
Traffic Light Bounding Box
    |
    v
LiDAR Point Cloud
    |
    v
LiDAR to Camera Projection
    |
    v
Projected LiDAR Points
    |
    v
Bounding Box Association
    |
    v
Near-Point Selection
    |
    v
3D Distance Estimation
    |
    v
LiDAR–Camera Fusion
    |
    v
Traffic Light Decision

## LiDAR–Camera Calibration

The system uses calibrated camera intrinsic and LiDAR–camera extrinsic parameters to transform LiDAR points into the camera coordinate frame and project them onto the camera image.

The fusion pipeline performs the following steps:

1. Transform LiDAR points into the camera coordinate frame using the extrinsic transformation.
2. Project the transformed 3D points onto the camera image using the camera intrinsic matrix.
3. Remove points behind the camera.
4. Remove projected points outside the image boundaries.
5. Associate valid projected LiDAR points with YOLO detection bounding boxes.

The projection process can be summarized as:

LiDAR Point (X, Y, Z)
    |
    v
Extrinsic Transformation
    |
    v
Camera Coordinate Frame
    |
    v
Camera Intrinsic Projection
    |
    v
Image Pixel (u, v)
    |
    v
YOLO Bounding Box Association

## YOLO-Based Traffic Light Detection

A YOLO-based object detection model is used to detect traffic lights from the camera image.

For each detected traffic light, the system extracts:

- Bounding box coordinates
- Object class
- Projected LiDAR points inside the bounding box
- LiDAR-based distance information

The YOLO model is configured to use CUDA when a compatible GPU is available.

## LiDAR Depth Association

After projecting the LiDAR point cloud onto the camera image, the system identifies LiDAR points that fall inside each detected traffic-light bounding box.

An adaptive filtering approach is used to select the points corresponding to the nearest part of the detected object.

The process is:

1. Collect LiDAR points inside the detection bounding box.
2. Calculate the Euclidean distance of each point.
3. Find the minimum distance.
4. Select points within 5% of the minimum distance.
5. If fewer than three points are available, select up to the five closest points.
6. Calculate the centroid of the selected points.
7. Use the selected points for depth and spatial estimation.

This provides a 3D LiDAR-based distance estimate for the camera-detected traffic light.

## Distance Estimation

For each detected traffic light, the system calculates:

- Minimum LiDAR distance
- Mean distance of selected LiDAR points
- Standard deviation
- LiDAR-associated distance
- Difference between the estimated distances

The detected traffic-light label and corresponding distance are published through ROS.

Example:

    red: 8.4200
    green: 14.2100

## Traffic Light Decision Making

The fused traffic-light information is used to generate a decision signal for autonomous navigation.

The current implementation uses a trigger distance of:

    10.5 m

When a red traffic light is detected within the trigger distance, the system publishes a TRUE decision.

When a green traffic light is detected, the system publishes a FALSE decision.

A timeout mechanism is also implemented to reset the decision state when a relevant traffic-light detection is no longer observed.

Decision logic:

Traffic Light Detection
    |
    v
Distance Check
    |
    +----------------------+
    |                      |
    v                      v
> 10.5 m                <= 10.5 m
    |                      |
    |                      v
    |                 Red Detected
    |                      |
    |                      v
    |                    TRUE
    |
    v
Continue

Green Detected
    |
    v
FALSE

## ROS Architecture

### Subscribed Topics

| Topic | Message Type | Purpose |
|---|---|---|
| `/image_raw` | `sensor_msgs/Image` | Camera image |
| `/obstacle_detector/cloud_clusters` | `sensor_msgs/PointCloud2` | LiDAR point cloud |
| `/obstacle_detector/jsk_bboxes` | `BoundingBoxArray` | LiDAR/obstacle bounding boxes |
| `/ndt_pose` | `PoseStamped` | Vehicle localization pose |

### Published Topics

| Topic | Purpose |
|---|---|
| `/lidar_projection/image` | LiDAR points projected onto camera image |
| `/classified_image/image` | Image with detected traffic-light information |
| `/classified_lidar/pointcloud` | LiDAR points associated with detected objects |
| `/classified_lidar/labels` | RViz visualization markers |
| `/lcf_tl_distance` | Detected traffic-light label and estimated distance |
| `/lcf_tl_decision` | Traffic-light decision state |

## Visualization

The system provides visualization of:

- LiDAR points projected onto the camera image
- YOLO detection bounding boxes
- Traffic-light class labels
- Estimated object distances
- Classified LiDAR points
- 3D labels in the LiDAR frame

The projected image can be published for camera-side visualization, while classified LiDAR points and object labels can be visualized in RViz.

## Key Features

- Real-time ROS-based sensor fusion
- LiDAR–Camera extrinsic calibration
- 3D LiDAR to 2D image projection
- YOLO-based traffic-light detection
- LiDAR-based depth estimation
- Camera–LiDAR object association
- Adaptive nearest-point selection
- Object-wise distance estimation
- Classified LiDAR point-cloud generation
- RViz visualization
- Distance-based traffic-light decision making
- Automatic decision timeout handling

## Technology

- ROS
- Python
- OpenCV
- PyTorch
- Ultralytics YOLO
- NumPy
- Shapely
- LiDAR
- Camera
- Point Cloud Processing
- Sensor Fusion
- Computer Vision
- Autonomous Navigation

## Requirements

- ROS
- Python 3
- OpenCV
- PyTorch
- Ultralytics
- NumPy
- Shapely
- sensor_msgs
- cv_bridge
- jsk_recognition_msgs
- visualization_msgs

## Project Structure

    lidar-camera-fusion-traffic-light/
    |
    +-- README.md
    |
    +-- src/
    |   +-- lcf_tl.py
    |
    +-- weights/
    |   +-- traffic_lights.pt
    |
    +-- config/
    |   +-- calibration.yaml
    |
    +-- results/
        +-- visualization/

The actual implementation and trained model weights are proprietary and are not included in this public repository.

## Code Availability

The implementation is part of a proprietary autonomous navigation system and the complete source code is therefore not publicly available.

This repository documents the LiDAR–Camera fusion architecture, perception pipeline, depth-estimation methodology, and traffic-light decision-making approach.

## Author

S. Sathvik

Research Assistant

TiHAN – Technology Innovation Hub for Autonomous Navigation

IIT Hyderabad
