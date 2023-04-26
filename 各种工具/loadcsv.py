import pandas as pd
df = pd.DataFrame()

chunksize = 1e5    #这个数字设置多少有待考察
for chunk in pd.read_csv('E:\\DLcode\\dr5data\\dr5.csv', chunksize=chunksize):
    df = pd.concat([df, chunk])