import subprocess
variables = ["ZDN"]
n_amount = [0, 90, 180, 270]
cmds = list()
for varia in variables:
    for nn in n_amount:
        cmd = f"python F:\\paper_code\\wofost\\MCMC\\Wofost_DREAM_WLP_V01.py {varia}{nn} {nn}"
        cmds.append(cmd)

for cmd in cmds:
    subprocess.call(cmd, shell=True)
