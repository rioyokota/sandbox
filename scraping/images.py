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
    url_file = '2021-01-01.tsv.gz'
    save_path = '/groups1/gcc50533/data/twitter/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = pd.read_csv(url_file,header=None,sep='\t')
    total = len(f)
    df = pd.read_csv(url_file,header=None,sep='\t',chunksize=1,nrows=10)
    for count, line in enumerate(df):
        if line.values[0][6] != 'グランブルー ファンタジー':
            url = line.values[0][3]
            print(f'{count} / {total}')
            download(url,save_path)

if __name__ == '__main__':
    main()
