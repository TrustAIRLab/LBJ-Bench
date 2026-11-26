"""Download the parts of LIBERO-Safety's assets.zip that the benchmark needs, with HTTP range requests.

usage: python fetch_assets.py <LIBERO-Safety>/libero/libero        (folders and files are listed in UPSTREAM_ASSETS)"""
import concurrent.futures, io, os, struct, sys, urllib.request, zipfile, zlib
HERE = os.path.dirname(os.path.abspath(__file__))
URL = 'https://huggingface.co/datasets/LIBERO-Safety/libero_safety_assets/resolve/main/assets.zip'
CHUNK = 32 << 20
SLACK = 1024


def fetch(url, start, end):
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    for attempt in range(5):
        try:
            return urllib.request.urlopen(request, timeout=120).read()
        except OSError:
            if attempt == 4:
                raise


class Remote(io.RawIOBase):

    def __init__(self, url):
        reply = urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=0-0'}))
        (self.url, self.size, self.pos) = (reply.geturl(), int(reply.headers['Content-Range'].split('/')[-1]), 0)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size:
            return 0
        data = fetch(self.url, self.pos, min(self.size, self.pos + len(buffer)) - 1)
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def save(remote, group, dest):
    start = group[0].header_offset
    end = min(remote.size, max(i.header_offset + 30 + len(i.filename.encode()) + SLACK + i.compress_size for i in group)) - 1
    data = fetch(remote.url, start, end)
    for info in group:
        at = info.header_offset - start
        assert data[at:at + 4] == b'PK\x03\x04', info.filename
        (name_len, extra_len) = struct.unpack('<HH', data[at + 26:at + 30])
        raw = data[at + 30 + name_len + extra_len:][:info.compress_size]
        content = raw if info.compress_type == zipfile.ZIP_STORED else zlib.decompress(raw, -15)
        assert zlib.crc32(content) == info.CRC, info.filename
        path = os.path.join(dest, info.filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'wb').write(content)
    return sum(i.file_size for i in group)


def main():
    dest = sys.argv[1]
    wanted = ['assets/' + line.strip().strip('/') for line in open(os.path.join(HERE, 'UPSTREAM_ASSETS')) if line.strip()]
    inside = lambda name, w: name == w or name.startswith(w + '/')
    remote = Remote(URL)
    files = [i for i in zipfile.ZipFile(io.BufferedReader(remote, 1 << 20)).filelist if (not i.is_dir()) and any(inside(i.filename, w) for w in wanted)]
    found = {w for w in wanted if any(inside(i.filename, w) for i in files)}
    if found != set(wanted):
        raise SystemExit(f'not in assets.zip: {sorted(set(wanted) - found)}')
    files = sorted((i for i in files if not (os.path.exists(os.path.join(dest, i.filename)) and os.path.getsize(os.path.join(dest, i.filename)) == i.file_size)), key=lambda i: i.header_offset)
    groups = []
    for info in files:
        if groups and info.header_offset - groups[-1][0].header_offset < CHUNK and info.header_offset - (groups[-1][-1].header_offset + groups[-1][-1].compress_size) < (1 << 20):
            groups[-1].append(info)
        else:
            groups.append([info])
    print(f'{len(files)} files, {sum(i.file_size for i in files) / 1e6:.0f} MB to download', flush=True)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        done = sum(pool.map(lambda g: save(remote, g, dest), groups))
    print(f'downloaded {done / 1e6:.0f} MB')


if __name__ == '__main__':
    main()
