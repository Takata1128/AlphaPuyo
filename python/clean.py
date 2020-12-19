import os
import shutil
from define import DATA_PATH

if __name__ == '__main__':
    if os.path.exists(DATA_PATH+'/'):
        shutil.rmtree(DATA_PATH+'/')
