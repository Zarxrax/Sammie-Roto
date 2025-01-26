import os
import requests
import hashlib
from tqdm import tqdm


_links = [
    ('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt', '2b30654b6112c42a115563c638d238d9'),
    ('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt', 'ec7bd7d23d280d5e3cfa45984c02eda5'),
    ('https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_s_512x512.pt', '962e151a9dca3b75d8228a16e5264010'),
]

def download_models():
    os.makedirs('checkpoints', exist_ok=True)
    for link, md5 in _links:
        # download file if not exists with a progressbar
        filename = link.split('/')[-1]
        if not os.path.exists(os.path.join('checkpoints', filename)) or hashlib.md5(open(os.path.join('checkpoints', filename), 'rb').read()).hexdigest() != md5:
            print(f'Downloading {filename}...')
            r = requests.get(link, stream=True)
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024
            t = tqdm(total=total_size, unit='iB', unit_scale=True)
            with open(os.path.join('checkpoints', filename), 'wb') as f:
                for data in r.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()
            if total_size != 0 and t.n != total_size:
                raise RuntimeError('Error while downloading %s' % filename)
        else:
            print(f'{filename} already downloaded.')
            

if __name__ == '__main__':
    download_models()