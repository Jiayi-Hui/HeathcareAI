#!/bin/bash

# Clone UniEval repository if not already present
if [ ! -d "UniEval" ]; then
    git clone https://github.com/maszhongming/UniEval.git
fi

# Navigate to UniEval directory and install requirements
cd UniEval
pip install -r requirements.txt

# Return to root directory
cd ..
