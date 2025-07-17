"""easy_to_hard_data.py
Python package with datasets for studying generalization from
    easy training data to hard test examples.
Developed as part of easy-to-hard (github.com/aks2203/easy-to-hard).
Avi Schwarzschild
June 2021
"""

import errno
import os
import os.path
import tarfile
import urllib.request as ur

from tqdm import tqdm

GBFACTOR = float(1 << 30)


def extract_zip(path, folder):
    file = tarfile.open(path)
    file.extractall(folder)
    file.close


def download_url(url, folder):
    filename = url.rpartition("/")[2]
    path = os.path.join(folder, filename)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        print("Using existing file", filename)
        return path
    print("Downloading", url)
    makedirs(folder)
    # track downloads
    ur.urlopen(f"http://avi.koplon.com/hit_counter.py?next={url}")
    data = ur.urlopen(url)
    size = int(data.info()["Content-Length"])
    chunk_size = 1024 * 1024
    num_iter = int(size / chunk_size) + 2

    downloaded_size = 0

    try:
        with open(path, "wb") as f:
            pbar = tqdm(range(num_iter))
            for i in pbar:
                chunk = data.read(chunk_size)
                downloaded_size += len(chunk)
                pbar.set_description(
                    "Downloaded {:.2f} GB".format(float(downloaded_size) / GBFACTOR)
                )
                f.write(chunk)
    except:
        if os.path.exists(path):
            os.remove(path)
        raise RuntimeError("Stopped downloading due to interruption.")

    return path


def makedirs(path):
    try:
        os.makedirs(os.path.expanduser(os.path.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise e
