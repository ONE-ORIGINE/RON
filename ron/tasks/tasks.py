from __future__ import annotations
import torch
from torch.utils.data import Dataset

MASK_NONE = 0
MASK_X = 1
MASK_Y = 2
MASK_Z = 3
MASK_XY = 4
MASK_YZ = 5
MASK_XZ = 6

class TriAxisDataset(Dataset):
    """Tri-axial benchmark.

    Modes:
    - triaxial: fixed full triplet.
    - causal_consequence: Z depends on X,Y.
    - bridge_present: Y bridges X,Z.
    - tri_masked: every example has a known/missing pattern; the model must infer the missing axis from the same RON geometry.
    """
    def __init__(self,n:int,vocab:int=12,seed:int=123,mode:str='triaxial'):
        g=torch.Generator().manual_seed(seed+707)
        self.n=n; self.vocab=vocab; self.mode=mode
        X=torch.randint(1,vocab,(n,),generator=g)
        Y=torch.randint(1,vocab,(n,),generator=g)
        Z=torch.randint(1,vocab,(n,),generator=g)
        if mode=='causal_consequence':
            Z=(2*X+Y+torch.randint(0,4,(n,),generator=g))%vocab
            Z=Z.clamp_min(1)
        if mode=='bridge_present':
            Y=(X+Z+torch.randint(0,3,(n,),generator=g))%vocab
            Y=Y.clamp_min(1)
        self.triplet=torch.stack([X,Y,Z],1)

        if mode == 'tri_masked':
            masks = torch.tensor([MASK_NONE, MASK_X, MASK_Y, MASK_Z, MASK_XY, MASK_YZ, MASK_XZ], dtype=torch.long)
            self.mask = masks[torch.arange(n) % len(masks)]
            perm = torch.randperm(n, generator=g)
            self.mask = self.mask[perm]
        else:
            self.mask = torch.zeros(n, dtype=torch.long)

        self.past=((Y-X).abs()+X)%6
        self.future=((Z-Y).abs()+Z)%6
        self.present=((X+Y+Z)%6)
        self.cause=((2*X+Y+Z)%6)
        self.reverse=((X+2*Y+3*Z)%6)
        self.consequence=((3*X+Y+2*Z+(X-Y).abs())%6)
        self.diagonal=((self.past+self.future+self.cause+self.present)%6)

    def __len__(self):
        return self.n

    def __getitem__(self,i):
        return {k:getattr(self,k)[i].long() for k in ['triplet','mask','past','future','present','cause','reverse','consequence','diagonal']}

class VisionAxisDataset(Dataset):
    """Small functional vision benchmark with cached image tensors.

    The codex in the model is deterministic; keeping the input small and cached avoids
    wasting time on repeated construction of synthetic images.
    """
    def __init__(self,n:int,image_size:int=12,seed:int=123,mode:str='vision_orientation'):
        g=torch.Generator().manual_seed(seed+909); H=W=image_size
        imgs=[]; prevs=[]; depths=[]; labels=[]
        xx=torch.linspace(-1,1,W).view(1,W).expand(H,W)
        yy=torch.linspace(-1,1,H).view(H,1).expand(H,W)
        for i in range(n):
            c=int(torch.randint(0,6,(1,),generator=g))
            img=torch.zeros(1,H,W); prev=torch.zeros(1,H,W); dep=torch.zeros(1,H,W)
            if c==0:
                img[:,:,W//3]=1; prev[:,:,max(0,W//3-1)]=1
            elif c==1:
                img[:,:,2*W//3]=1; prev[:,:,min(W-1,2*W//3+1)]=1
            elif c==2:
                img[:,H//3,:]=1; prev[:,max(0,H//3-1),:]=1
            elif c==3:
                img[:,2*H//3,:]=1; prev[:,min(H-1,2*H//3+1),:]=1
            elif c==4:
                for k in range(H):
                    img[:,k,k]=1; prev[:,k,max(0,k-1)]=1
                dep[0]=xx
            else:
                for k in range(H):
                    img[:,k,W-1-k]=1; prev[:,k,min(W-1,W-k)]=1
                dep[0]=-xx
            if mode=='vision_depth':
                dep[0]=(xx+yy).clamp(-1,1) if c%2==0 else (xx-yy).clamp(-1,1)
            if mode=='vision_intersection':
                img[:,H//2,:]=1; img[:,:,W//2]=1
                c=4 if c%2==0 else 5
            img=(img+0.015*torch.rand(img.shape,generator=g)).clamp(0,1)
            imgs.append(img); prevs.append(prev); depths.append(dep); labels.append(c)
        self.image=torch.stack(imgs)
        self.prev_image=torch.stack(prevs)
        self.depth=torch.stack(depths)
        f=torch.tensor(labels).long()
        self.future=f
        self.past=(f+3)%6
        self.present=(f+1)%6
        self.cause=(2*f+1)%6
        self.reverse=(f+2)%6
        self.consequence=(3*f+2)%6
        self.diagonal=(f+self.past+self.cause)%6
        self.mask=torch.zeros(n,dtype=torch.long)

    def __len__(self):
        return len(self.future)

    def __getitem__(self,i):
        return {k:getattr(self,k)[i] for k in ['image','prev_image','depth','mask','past','future','present','cause','reverse','consequence','diagonal']}
