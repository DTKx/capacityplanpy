import os
import sys

# print(__file__)
# print(os.path.dirname(__file__))
# print(os.path.abspath(os.getcwd()))
# print(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# print(os.path.abspath(os.path.join("..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))
import capacityplanpy