import pandas
import dask.dataframe as dd
import numpy as np
import os
from tqdm.auto import tqdm

CL = ('A', 'B', 'F', 'G', 'K', 'M', 'O')
scope = {}

for i in np.arange(0,6):
    strname = 'E:\\DLcode\\tools\\{}_SNU_STAR.csv'.format(CL[i])
    exec("{} = dd.read_csv(strname)".format(CL[i]))
    exec("{} = {}.compute()[0:1000]".format(CL[i],CL[i]))
    exec("{}['obsdate'] = {}['obsdate'].astype(str)".format(CL[i],CL[i]))
    exec("{}['spid'] = {}['spid'].astype(str)".format(CL[i], CL[i]))
    exec("{}['fiberid'] = {}['fiberid'].astype(str)".format(CL[i], CL[i]))
    exec("{}['obsdate'] = {}['obsdate'].str.slice(0,4)+{}['obsdate'].str.slice(5,7)+{}['obsdate'].str.slice(8,10)".format(CL[i],CL[i],CL[i],CL[i]))
    exec("{}.loc[:,'link'] = 'http://dr5.lamost.org/v3/sas/fits/'+{}['obsdate']+'/'+{}['planid'].astype(str)+'/'+'spec-'+{}['lmjd'].astype(str)+'-'+{}['planid'].astype(str)+'_sp'+{}['spid'].str.zfill(2)+'-'+{}['fiberid'].str.zfill(3)+'.fits.gz'".format(CL[i],CL[i],CL[i],CL[i],CL[i],CL[i],CL[i]))
    exec("{}.to_csv('{}_SNU_link.csv')".format(CL[i], CL[i]))
    exec('del {}'.format(CL[i]))
    print('done')
print('done all')