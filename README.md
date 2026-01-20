# Gravity Panda


## Getting started

To run this project you need to clone the repository and setup a virtual environment containing the packages from the pyproject.toml file. For this you can run ```uv run main.py```.

## Description
In this project we want a pandas robot arm to catch a ball with a cup attached to one of its gripper "fingers".

These are the args that can be added when running the project:
```
#usage: main.py [--botop] [--real | --sim-only] [--input INPUT] [--display]

# options:
#  --botop        Use botop in addition to the simulator for a comparison.
#  --real         Use the real robot (must be used with --botop)
#  --input INPUT  Select camera: wrist, top or standalone
#  --display      Show camera view
#  --sim-only     Only run simulation using precomputed trajectory

```
For now, run: ```uv run main.py --input top --display``` or ```python main.py --input top --display```

## Visuals
Tools like ttygif can help, but check out Asciinema for a more sophisticated method.


## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.


## Authors
James Reynaldo, Phil Garweg and Stephanie Day
