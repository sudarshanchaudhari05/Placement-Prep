import sys

with open(r"C:\Users\sudar\Documents\Collage\Placement prep\Calculator\codesys_info.txt", "w") as f:
    f.write("Python version: " + sys.version + "\n")
    try:
        import scriptengine
        f.write("scriptengine imported successfully\n")
    except Exception as e:
        f.write("scriptengine error: " + str(e) + "\n")
    try:
        f.write("projects dir: " + str(dir(projects)) + "\n")
    except Exception as e:
        f.write("projects error: " + str(e) + "\n")
