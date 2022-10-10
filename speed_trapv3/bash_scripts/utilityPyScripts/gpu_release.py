# run apt-get install lsof
import os

os.system("lsof /dev/nvidia* | awk '{print $2}' | xargs -I {} kill {}")
