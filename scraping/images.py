import os
import pandas as pd
import requests
from timeit import default_timer as timer

def download(url, pathname):
    data = requests.get(url).content
    filename = os.path.join(pathname, url.split("/")[-1])
    with open(filename, "wb") as f:
        f.write(data)
    if os.path.getsize(filename) == 0:
        os.remove(filename)

def main():
    date = '2021-01-01'
    url_file = 'url/' + date + '.tsv.gz'
    save_path = '/groups1/gcc50533/data/twitter/' + date + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = pd.read_csv(url_file,header=None,sep='\t')
    total = len(f)
    df = pd.read_csv(url_file,header=None,sep='\t',chunksize=1) #,nrows=10)
    tic = timer()
    for count, line in enumerate(df):
        if count > 433429 and line.values[0][6] != 'グランブルー ファンタジー':
            url = line.values[0][3]
            download(url,save_path)
        toc = timer()
        print(f'{count} / {total} : {toc-tic} s')

if __name__ == '__main__':
    main()
