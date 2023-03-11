import subprocess
import sys

print("Python version: ", sys.version)
print("Python version info: ", sys.version_info)
 
print("Python version info[0]: ", sys.version_info[0])
print("Python version info[1]: ", sys.version_info[1])
print("Python version info[2]: ", sys.version_info[2])
print("Python version info[3]: ", sys.version_info[3])
print("Python version info[4]: ", sys.version_info[4])

cmds = ["conda.bat activate drawing & python ./test.py ZDN90"]
subprocess.check_call(cmds[0])
