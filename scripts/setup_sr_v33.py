import hashlib,json,sys
from pathlib import Path
import requests

root=Path('/root/autodl-tmp/liquid-depth-artifacts/pretrained/swinir_v33')
root.mkdir(parents=True,exist_ok=True)
session=requests.Session()
session.headers['User-Agent']='liquid-depth-research/1.0'
r=session.get('https://api.github.com/repos/JingyunLiang/SwinIR/commits/main',timeout=60);r.raise_for_status()
sha=r.json()['sha']
files={
    'network_swinir.py':f'https://raw.githubusercontent.com/JingyunLiang/SwinIR/{sha}/models/network_swinir.py',
    'LICENSE':f'https://raw.githubusercontent.com/JingyunLiang/SwinIR/{sha}/LICENSE',
    'swinir_x2.pth':'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth'}
provenance={}
for name,url in files.items():
    path=root/name
    if not path.exists():
        r=session.get(url,timeout=180);r.raise_for_status()
        if len(r.content)<100:raise RuntimeError('invalid_download')
        path.write_bytes(r.content)
    provenance[name]=dict(url=url,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
(root/'provenance.json').write_text(json.dumps(dict(commit=sha,files=provenance),indent=2))
sys.path.insert(0,str(root))
import torch
from network_swinir import SwinIR
model=SwinIR(upscale=2,in_chans=3,img_size=64,window_size=8,img_range=1.,depths=[6]*4,embed_dim=60,
             num_heads=[6]*4,mlp_ratio=2,upsampler='pixelshuffledirect',resi_connection='1conv')
weights=torch.load(root/'swinir_x2.pth',map_location='cpu',weights_only=True)
model.load_state_dict(weights['params'] if 'params' in weights else weights,strict=True)
model.eval().cuda()
with torch.inference_mode():out=model(torch.zeros(1,3,128,160,device='cuda'))
print('VERIFIED',out.shape,provenance,flush=True)
