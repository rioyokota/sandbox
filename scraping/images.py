import pandas as pd
import requests
import os

def download(url, pathname):
    data = requests.get(url).content
    filename = os.path.join(pathname, url.split("/")[-1])
    with open(filename, "wb") as f:
        f.write(data)
    if os.path.getsize(filename) == 0:
        os.remove(filename)

def main():
    df = pd.read_csv('2021-01-01.tsv.gz',header=None,sep='\t',chunksize=1,nrows=20)
    for line in df:
        if line.values[0][6] != 'グランブルー ファンタジー':
            url = line.values[0][3]
            print(line.values[0][0])
            download(url,'data/')

if __name__ == '__main__':
    main()
