# Toolkit-Interpretability-in-Reinforcement-Learning

Toolkit Interpretability in Reinforcement Learning is a Python program to explain model based MDPs.

## Installation

Install any neccessary dependencies:
Numpy
Matplotlib

## Usage

Import the explanations package
Initialize the problem
Use any of the possible explanations

```python
from explanations import *

expl = Explanations(100,Pl,Rl,4,0.90,[99]) #example of initialization
expl.single_get_state_explanation_distribution(0,labels=['Action UP','Action DOWN','Action LEFT','Action RIGHT'])#example of function
```
The Test.py file has a number of examples and tests to run.

For more information consult the explanation.py file.

## Contributing
No contributions are necessary, if any problem arises open an issue

## Authors
This project was made by Renato Paulo da Costa Cardoso.