import cv2 as cv
from cv2 import aruco
import numpy as np
import robotic as ry

def rt_to_T(R, t):
    T = np.eye(4, dtype=float)
    T[:3,:3] = R
    T[:3, 3] = t.reshape(3)
    return T

def rvec_tvec_to_T(rvec, tvec):
    R, _ = cv.Rodrigues(rvec)
    return rt_to_T(R, tvec)     
  
def matrix2vector(T):
   vec = np.zeros(7) 
   R = T[:3, :3]
   quat = ry.Quaternion().setMatrix(R).asArr() 
   vec[:3] = T[:3, 3]
   vec[3:] = quat
   return vec  

def detect_aruco_markers(rgb, K, dist, W, H):
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
    params = aruco.DetectorParameters()

    out = rgb.copy()
    detection_txt = "Poses (cam):"
    T_world2cam = np.array([[-1, 0, 0, 0],[0, 1, 0, 0.45],[0, 0, -1, 2.30],[0, 0, 0, 1]])       #world coordinates of the virtual top down camera. Will not work for other inputs!
    T_world2markerX = np.array([[1, 0, 0, 0],[0, 1, 0, 0.45],[0, 0, 1, 0.65],[0, 0, 0, 1]])         #world coordinates of marker 100
    T_cam2marker100 = np.eye(4)
    marker_length = 0.092  # 0.05 would be standard for academic projects

    #### BEGIN SOLUTION for 1.1

    object_points = np.array([
        [-marker_length/2,  marker_length/2, 0],
        [ marker_length/2,  marker_length/2, 0],
        [ marker_length/2, -marker_length/2, 0],
        [-marker_length/2, -marker_length/2, 0]
    ], dtype=np.float32)

    # Detect markers
    detector = aruco.ArucoDetector(dictionary, params)
    marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(out)

    if marker_corners is None or marker_ids is None:
        print("No markers detected.")
        return
    
    # Build marker dict
    marker_dict = {int(m_id[0]): {"xyxy": corners} for m_id, corners in zip(marker_ids, marker_corners)}
    
    # Compute and draw poses  
    for marker_id, marker_data in marker_dict.items():
        corners = marker_data.get('xyxy')
        retval,rvec,tvec = cv.solvePnP(objectPoints=object_points, imagePoints=corners, cameraMatrix=K, distCoeffs=dist)
        if retval:
            marker_dict.get(marker_id)["pnp"] = (rvec,tvec)
            cv.drawFrameAxes(out, K, dist, rvec, tvec, 0.05)

    # Draw marker bboxes 
    aruco.drawDetectedMarkers(out, marker_corners, marker_ids)
    cv.putText(out, detection_txt, (3,10), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1, cv.LINE_AA)
    cv.imshow("ArUco Markers", out)
    
    # World to camera
    print("Using 'T_world2cam' to compute marker world coordinates:")
    T_flipY = np.diag([1, -1, 1, 1])
    for marker_id, marker_data in marker_dict.items():
        rvec, tvec  = marker_data.get('pnp')
        T_c_m = rvec_tvec_to_T(rvec, tvec)
        T_c_m_flip = T_flipY @ T_c_m
        T_w_m  = T_world2cam @ T_c_m_flip
        pose = matrix2vector(T_w_m)

        marker_data['T_c_m_w2c'] = T_c_m
        marker_data['T_w_m_w2c'] =  T_w_m
        marker_data['pose_w2c'] = pose

        print(f"Marker {marker_id} Pose: {np.round(pose, 2)}")

    # World to marker
    print("Using 'T_world2marker100' to compute marker world coordinates:")

    
    
    for marker_id, marker_data in marker_dict.items():
        T_m_c = np.linalg.inv(marker_data.get('T_c_m_w2c'))
        T_cam2marker = T_world2markerX @ T_m_c
        T_c_m = marker_data.get('T_c_m_w2c')
        T_c_m_flip = T_flipY @ T_c_m
        T_w_m  = T_cam2marker @ T_c_m_flip
        pose = matrix2vector(T_w_m)

        marker_data['T_c_m_w2m'] = T_c_m
        marker_data['T_w_m_w2m'] =  T_w_m
        marker_data['pose_w2m'] = pose
        print(f"Marker {marker_id} Pose: {np.round(pose, 2)}")
                         

    cv.putText(out, detection_txt, (3,10), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1, cv.LINE_AA)
    cv.imshow("ArUco Markers", out)
    cv.waitKey(-1)
    cv.destroyAllWindows() 