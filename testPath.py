import os

# print(os.getcwd())  # C:\Users\Debs\Documents\01_UFU_local\01_comp_evolutiva\capacityplanpy

# print(os.path.dirname(os.getcwd()))  # C:\Users\Debs\Documents\01_UFU_local\01_comp_evolutiva

# print(__file__)  # c:\Users\Debs\Documents\01_UFU_local\01_comp_evolutiva\capacityplanpy\testPath.py

# print(
#     os.path.dirname(__file__)
# )  # c:\Users\Debs\Documents\01_UFU_local\01_comp_evolutiva\capacityplanpy

# print(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# )  # c:\Users\Debs\Documents\01_UFU_local\01_comp_evolutiva\capacityplanpy

# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# print(os.path.dirname("output_raw"))  # c:\Users\Debs\Documents\01_UFU_local\01_comp_evolutiva

# dirname = os.path.dirname(__file__)
# filename = os.path.join(dirname, 'data\output_raw')
# print(filename)

print(os.path.abspath(os.path.join(os.path.dirname(__file__), "data/output_raw/")))
output_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/output_raw/"))
file_path = os.path.join(output_data_path, "cprofile.txt")
print(file_path)
print(os.getcwd())
print(os.path.dirname(os.getcwd()))

print(os.path.abspath(os.path.join(os.path.dirname(__file__), "data/input/")))
