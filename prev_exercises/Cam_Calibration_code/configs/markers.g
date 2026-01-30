## Additional camera to see all markers
cameraTopMiddle(world): { 
    pose: [0.32, 1.09, 2.05, 0.306, 0.95, 0.002, -0.055], 
    shape: marker, 
    size: [0.1], 
    focalLength: 0.895, 
    width: 1280, 
    height: 720, 
    zRange: [0.5, 100] 
}


## Markers

aruco100: { X: [0.0, 0.3, 0.66, 0, 0, 0, 1], shape: mesh, mesh: <../meshes/aruco_marker_100.ply>}
aruco153: { X: [0.2, 0.2, 0.66, 0, 0, 0, 1], shape: mesh, mesh: <../meshes/aruco_marker_153.ply>}
aruco493: { X: [0.0, 0.3, 0.65, 0, 0, 0, 1], shape: mesh, mesh: <../meshes/aruco_marker_493.ply>}
#aruco116: { X: [-0.2, 0.2, 0.66, 0, 0, 0, 1], shape: mesh, mesh: <../meshes/aruco_marker_116.ply>}
#aruco136: { X: [0.5, 0.6, 0.66, 0, 0, 0, 1], shape: mesh, mesh: <../meshes/aruco_marker_136.ply>}

