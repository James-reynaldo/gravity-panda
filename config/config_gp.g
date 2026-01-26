
cameraTopMiddle(world): { 
    pose: [0.0, 3, 1.2, 0.0, 0.0, 0.707, -0.707], 
    shape: marker, 
    size: [0.1], 
    focalLength: 0.895, 
    width: 1280, 
    height: 720, 
    zRange: [0.5, 100] 
}

ball: { 
    pose: [-1, 0.2, 2], 
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

q_home: { 
    pose: [0., 0.2, 1.0], 
    shape: marker, 
    size: [0.1]
}