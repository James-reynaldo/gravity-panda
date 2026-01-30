## [0.29, 1.42, 2.2, 0.0, 0.0, 0.94, -0.342]
## [0.0, 3, 1.2, 0.0, 0.0, 0.707, -0.707]
cameraTopMiddle(world): { 
    pose: [0.32, 1.09, 2.05, 0.306, 0.95, 0.002, -0.055], 
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

shoeboxObstacle(world): {
  pose: [-0.8, 0.0, 1, 0.7071, 0, 0, 0.7071],
  shape: box,
  size: [0.15, 0.10, 0.06],
  color: [0.6, 0.3, 0.1],
  contact: 1
}
