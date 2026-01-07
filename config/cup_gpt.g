## Cup geometry - 120mm diameter, 60mm deep, 4mm walls
## Dimensions in meters: diameter=0.12m, depth=0.06m, thickness=0.004m

# Outer cylinder (visual representation)
cup_outer: {
    shape: cylinder,
    size: [0.06, 0.06],  # [radius, height]
    color: [0.9, 0.9, 0.9, 0.8],
    contact: 0
}

# Inner cylinder (hollow part) - slightly smaller and positioned to create wall thickness
cup_inner: {
    shape: cylinder,
    size: [0.056, 0.061],  # [radius=60mm-4mm, height slightly more than outer]
    color: [0.9, 0.9, 0.9, 0.8],
    contact: 0,
    rel: [0, 0, 0.002, 1, 0, 0, 0]  # Slightly offset upward to expose bottom
}

# Bottom disc
cup_bottom: {
    shape: cylinder,
    size: [0.06, 0.004],  # [radius, thickness]
    color: [0.9, 0.9, 0.9],
    contact: 1,
    rel: [0, 0, -0.028, 1, 0, 0, 0]  # Positioned at bottom: -(depth/2 - thickness/2)
}

# Side walls collision (for contact simulation)
cup_wall: {
    shape: cylinder,
    size: [0.06, 0.06],
    contact: 1,
    rel: [0, 0, 0, 1, 0, 0, 0]
}