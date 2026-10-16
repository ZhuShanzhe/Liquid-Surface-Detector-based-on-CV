"""Download publicly offered semantic priors with provenance; no registration bypass."""
import hashlib
import json
import zipfile
from pathlib import Path

import requests

ROOT=Path('/root/autodl-tmp/liquid-depth-data/external/water-obstacle-priors-v2')
SOURCES={
    'images.zip':'https://data.vicos.si/borja/mastr1325_dataset/MaSTr1325_images_512x384.zip',
    'masks.zip':'https://data.vicos.si/borja/mastr1325_dataset/MaSTr1325_masks_512x384.zip',
}


def main():
    ROOT.mkdir(parents=True,exist_ok=True);manifest={'dataset':'MaSTr1325','license':'CC BY 4.0',
       'license_source':'https://box.vicos.si/borja/viamaro/index.html',
       'citation':'Bovcon, Muhovic, Pers, Kristan. The MaSTr1325 dataset for training deep USV obstacle detection models. IROS 2019.',
       'role':'RGB water/obstacle appearance prior only, not industrial liquid metric depth supervision',
       'label_limit':'Environment is not synonymous with floating object; only isolated water-surrounded components become weak positives.',
       'files':{}}
    for name,url in SOURCES.items():
        target=ROOT/name
        if not target.exists():
            with requests.get(url,stream=True,timeout=(20,120)) as response:
                response.raise_for_status()
                with target.open('xb') as f:
                    for chunk in response.iter_content(1024*1024):f.write(chunk)
        with zipfile.ZipFile(target) as z:
            bad=z.testzip()
            if bad:raise RuntimeError('bad_zip_member:'+bad)
        manifest['files'][name]={'url':url,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        print('DOWNLOADED',name,target.stat().st_size,flush=True)
    (ROOT/'provenance.json').write_text(json.dumps(manifest,indent=2))
    print('PRIORS_COMPLETE',flush=True)


if __name__=='__main__':main()
