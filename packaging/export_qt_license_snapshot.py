"""Archive official Qt WebEngine notices, not a binary-generated SBOM."""
import argparse, hashlib, html, json, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urljoin
BASE='https://doc-snapshots.qt.io/qtwebengine/'
INDEX=BASE+'qtwebengine-licensing.html'
def get(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'LiquidDepth-license-archive/1.0'}),timeout=25) as response:
                return response.read().decode('utf-8')
        except Exception:
            if attempt==2:raise
            time.sleep(.5)
def body(document):
    value=re.search(r'<body[^>]*>(.*)</body>',document,re.S|re.I).group(1)
    value=re.sub(r'<script\b[^>]*>.*?</script>','',value,flags=re.S|re.I)
    value=re.sub(r'<(?:link|img|iframe)\b[^>]*>','',value,flags=re.I)
    return re.sub(r'href="(qtwebengine-3rdparty-[^"]+\.html)"',r'href="#\1"',value)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();index=get(INDEX)
    names=sorted(set(re.findall(r'href="(qtwebengine-3rdparty-[^"]+\.html)"',index)))
    if len(names)<80:raise RuntimeError('license_index_incomplete')
    with ThreadPoolExecutor(max_workers=12) as pool:documents=list(pool.map(lambda n:get(urljoin(BASE,n)),names))
    sections=['<!doctype html><html><head><meta charset="utf-8"><title>Qt WebEngine notices</title>',
        '<style>body{font-family:Arial;max-width:1100px;margin:auto;padding:24px}pre{white-space:pre-wrap;word-break:break-word}section{border-top:2px solid #567;margin-top:24px}header,nav,.sidebar,.toc,.footer{display:none}</style></head><body>',
        '<h1>Qt WebEngine / Chromium 第三方许可证离线快照</h1>',
        '<p>Qt 官方 WebEngine 6.140.0 文档，与本次 PySide6_WebEngine 6.12.0.140 发行系列对应。不是二进制自动生成的 SBOM。</p>',
        '<p>Qt 文档版权归 The Qt Company Ltd.，文档许可 GFDL 1.3；第三方代码按各章节原许可。</p>',body(index)]
    records=[]
    for name,document in zip(names,documents):
        sections+=['<section id="'+html.escape(name)+'"><p>官方来源：<a href="'+urljoin(BASE,name)+'">'+urljoin(BASE,name)+'</a></p>',body(document),'</section>']
        records.append(dict(url=urljoin(BASE,name),sha256=hashlib.sha256(document.encode()).hexdigest()))
    sections.append('</body></html>');args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(sections),encoding='utf-8')
    args.output.with_suffix('.sources.json').write_text(json.dumps(dict(source=INDEX,component='QtWebEngine 6.140.0',snapshot_date='2026-10-21',pages=len(records),documents=records),indent=2),encoding='utf-8')
    print('QT_NOTICES_SAVED',len(records),args.output.stat().st_size,flush=True)
if __name__=='__main__':main()
