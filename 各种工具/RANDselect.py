import pandas as pd
import numpy as np
import os

CL = ('O', 'B','A','F','G','K','M')
scope = {}
pd
for i in np.arange(0,6):
    strname = 'E:\\DLcode\\tools\\{}_STAR.csv'.format(CL[i])
    exec("{} = pd.read_csv(strname)".format(CL[i]))
    exec("{}1 = {}.sample(11716)".format(CL[i], CL[i]))
    exec("{}1.to_csv('{}_s.csv')".format(CL[i], CL[i]))
for i in np.arange(0,6):
    print(eval('{}1.info'.format(CL[i])))