## [0.29, 1.42, 2.2, 0.0, 0.0, 0.94, -0.342]
## [0.0, 3, 1.2, 0.0, 0.0, 0.707, -0.707]
cameraTopMiddle(world): { 
    pose: [0.29, 1.42, 2.2, 0.0, 0.0, 0.94, -0.342], 
    shape: marker, 
    size: [0.1], 
    focalLength: 0.895, 
    width: 1280, 
    height: 720, 
    zRange: [0.5, 100] 
}

ball: { 
    pose: [-.02, 0.3, 2], 
    shape: sphere, 
    size: [0.0315], 
    color: [0, 0.33, 0.66],
     contact: 0, 
     mass: 0.01 
}
