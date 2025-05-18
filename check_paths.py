import os, sys
print("cwd:", os.getcwd())
print("src/fusion/AV_Fusion.py exists:", os.path.exists("src/fusion/AV_Fusion.py"))
print("sys.path:")
for p in sys.path: print("  ", p)
