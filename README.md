# Gravity Panda


## Getting started

To run this project you need to clone the repository and setup a virtual environment containing the packages from the pyproject.toml file. For this you can run ```uv run main.py --args```.

## Description
In this project we want a panda robot arm to catch a ball with a cup which it holds.

## Args:
--input, "Select camera: wrist, top or standalone."

--botop, "Use botop in addition to the simulator for a comparison."

--real, "Use this arg if real robot is used."

--display, "Display camera frames."

--detectonly, "Use only ball detector without moving the robot"

--rotate, "Rotate camera frame by 180deg."

## Run:
```uv run main.py --botop``` or ```python main.py --botop```


## Authors
James Reynaldo, Phil Garweg and Stephanie Day
