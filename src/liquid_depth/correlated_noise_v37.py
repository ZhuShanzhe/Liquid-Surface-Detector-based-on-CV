"""Gaussian fields preserving ensemble marginal variance, not per-image mean."""
import numpy as np

def fields(shape,frames,seed,block=8,spatial_fraction=.5,temporal_rho=.9):
    rng=np.random.default_rng(seed);h,w=shape
    bh,bw=(h+block-1)//block+2,(w+block-1)//block+2
    shared=rng.normal(size=(bh,bw));oy,ox=rng.integers(0,block,2)
    def expand(a):return np.repeat(np.repeat(a,block,0),block,1)[oy:oy+h,ox:ox+w]
    for _ in range(frames):
        iid=rng.normal(size=shape);fresh=rng.normal(size=(bh,bw))
        shared=temporal_rho*shared+np.sqrt(1-temporal_rho**2)*fresh
        yield dict(iid=iid,spatial=np.sqrt(1-spatial_fraction)*iid+np.sqrt(spatial_fraction)*expand(fresh),
            persistent=np.sqrt(1-spatial_fraction)*iid+np.sqrt(spatial_fraction)*expand(shared))
