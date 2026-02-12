## Double Table [0.32, 1.09, 2.05, 0.306, 0.95, 0.002, -0.055],
## Simulation [0.0, 3, 1.2, 0.0, 0.0, 0.707, -0.707]
## Single Table [0.006,1.032,1.361,-0.024,0.008,0.896,-0.443]
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

tower(world):{
  pose: [-0.2, 0.3, .85],
  shape: cylinder,
  size: [0.4, 0.05],
  color: [1, .9, .5],
  contact: 1,
  friction: 0.1
}