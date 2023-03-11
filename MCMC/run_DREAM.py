import subprocess
import sys
from tqdm import tqdm
import time
tabName = ["ZDN0", "ZDN90", "ZDN180", "ZDN270"]
n_amount = [0, 90, 180, 270]
cmds = ["conda.bat activate pcse & python ./Wofost_DREAM_WLP.py %s %s" % (i, j) for i, j in zip(tabName, n_amount)]
time.sleep(3600)
for cmd in tqdm(cmds):
    try:
        print(cmd)
        subprocess.call(cmd, shell=False)
        print(f"{cmd} is done")
    except:
        continue
