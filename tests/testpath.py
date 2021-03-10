# To give the individual tests import context, create a tests/context.py file:
import os
import sys

# https://docs.python-guide.org/writing/structure/
print(__file__)  # c:\Users\Debs\Documents\02_python\09_tips\01_relative_paths\example_user\test.py
print(
    os.path.dirname(__file__)
)  # c:\Users\Debs\Documents\02_python\09_tips\01_relative_paths\example_user

# print(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import sample