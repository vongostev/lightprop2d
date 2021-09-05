# Generate documentation using https://github.com/vongostev/markdoc module
import os
import sys

sys.path.append(os.path.abspath('../../markdoc'))

from markdoc import MarkDoc

MarkDoc('../lightprop2d/beam2d.py', './Reference.md')
