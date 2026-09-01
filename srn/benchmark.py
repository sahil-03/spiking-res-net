"""Reproducible CIFAR-10 benchmark CLI."""
from __future__ import annotations

import argparse, json, os, random, time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import wandb
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .models import build_model, model_stats

MEAN=(0.4914,0.4822,0.4465); STD=(0.2470,0.2435,0.2616)
@dataclass
class Config:
    model: str="ann"; seed: int=0; width: int=16; blocks: int=3; tables: int=4; bits: int=4
    epochs: int=100; batch_size: int=128; lr: float=0.1; weight_decay: float=5e-4
    workers: int=4; data_dir: str="srn/data"; output_dir: str="outputs/run"
    max_train: int=0; max_val: int=0; max_steps: int=0; strong_aug: bool=False; final_test: bool=False; synthetic: bool=False
    hf_data: str="/persist/session-ad147782/data/cifar10-hf"

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def loaders(c: Config):
    if c.synthetic:
        ntr=c.max_train or 32; nva=c.max_val or 16
        g=torch.Generator().manual_seed(c.seed)
        # Deterministic class-coded images make this an optimization/overfit check.
        ytr=torch.arange(ntr)%10; yva=torch.arange(nva)%10
        prototypes=torch.randn(10,3,32,32,generator=g)
        tr=torch.utils.data.TensorDataset(prototypes[ytr]+.05*torch.randn(ntr,3,32,32,generator=g),ytr)
        va=torch.utils.data.TensorDataset(prototypes[yva]+.05*torch.randn(nva,3,32,32,generator=g),yva)
        kw=dict(batch_size=c.batch_size,num_workers=0,pin_memory=True)
        return DataLoader(tr,shuffle=True,**kw),DataLoader(va,shuffle=False,**kw),None
    base=[transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip()]
    if c.strong_aug: base += [transforms.RandAugment(num_ops=2,magnitude=9)]
    train_tf=transforms.Compose(base+[transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    eval_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    from .hf_data import HFCIFAR10
    raw=HFCIFAR10(c.hf_data,train=True)
    # Fixed stratified split without an external sklearn dependency.
    rng=np.random.default_rng(2026); labels=np.asarray(raw.targets); tr_parts=[]; va_parts=[]
    for cls in range(10):
        cls_idx=np.flatnonzero(labels==cls); rng.shuffle(cls_idx); va_parts.append(cls_idx[:500]); tr_parts.append(cls_idx[500:])
    tr=np.concatenate(tr_parts); va=np.concatenate(va_parts); rng.shuffle(tr); rng.shuffle(va)
    tr=tr[:c.max_train] if c.max_train else tr; va=va[:c.max_val] if c.max_val else va
    train=Subset(HFCIFAR10(c.hf_data,train=True,transform=train_tf),tr)
    val=Subset(HFCIFAR10(c.hf_data,train=True,transform=eval_tf),va)
    test=HFCIFAR10(c.hf_data,train=False,transform=eval_tf) if c.final_test else None
    kw=dict(batch_size=c.batch_size,num_workers=c.workers,pin_memory=True,persistent_workers=c.workers>0)
    return DataLoader(train,shuffle=True,**kw),DataLoader(val,shuffle=False,**kw),(DataLoader(test,shuffle=False,**kw) if test else None)

def metrics(model, loader, device):
    model.eval(); total=correct=nll=0.; conf=[]; hits=[]
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True); z=model(x)
            total+=y.numel(); correct+=(z.argmax(1)==y).sum().item(); nll+=F.cross_entropy(z,y,reduction="sum").item()
            p=z.softmax(1); conf.extend(p.max(1).values.cpu().tolist()); hits.extend((z.argmax(1)==y).cpu().tolist())
    conf=np.asarray(conf); hits=np.asarray(hits); ece=0.
    for lo in np.linspace(0,1,16)[:-1]:
        m=(conf>=lo)&(conf<lo+1/15); ece += m.mean()*abs(hits[m].mean()-conf[m].mean()) if m.any() else 0
    return {"accuracy":correct/total,"nll":nll/total,"ece":float(ece)}

def profile(model, device, batch=1):
    model.eval(); x=torch.randn(batch,3,32,32,device=device)
    for _ in range(10): model(x)
    if device.type=="cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t=time.perf_counter()
    for _ in range(50): model(x)
    if device.type=="cuda": torch.cuda.synchronize()
    elapsed=time.perf_counter()-t
    out={"latency_ms":elapsed/50*1000,"examples_per_s":batch*50/elapsed}
    if device.type=="cuda": out["peak_inference_bytes"]=torch.cuda.max_memory_allocated()
    out.update(model.instrumentation()); return out

def run(c: Config):
    print("RESOLVED_CONFIG="+json.dumps(asdict(c),sort_keys=True),flush=True)
    assert c.epochs>0 and c.batch_size>0
    seed_all(c.seed); out=Path(c.output_dir); out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train,val,test=loaders(c); model=build_model(c.model,width=c.width,blocks=c.blocks,tables=c.tables,bits=c.bits,seed=c.seed).to(device)
    stats=model_stats(model); print("MODEL_STATS="+json.dumps(stats),flush=True)
    wb=wandb.init(project="manifesto-resnet",name=f"{c.model}-seed{c.seed}",dir=str(out),mode=os.getenv("WANDB_MODE","offline"),config=asdict(c))
    opt=torch.optim.SGD(model.parameters(),lr=c.lr,momentum=.9,weight_decay=c.weight_decay,nesterov=True)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,c.epochs); scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    best=-1.; global_step=0; history=[]
    for ep in range(c.epochs):
        model.train(); seen=correct=loss_sum=0
        if device.type=="cuda": torch.cuda.reset_peak_memory_stats()
        for x,y in train:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type,enabled=device.type=="cuda"): z=model(x); loss=F.cross_entropy(z,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            seen+=y.numel(); correct+=(z.argmax(1)==y).sum().item(); loss_sum+=loss.item()*y.numel(); global_step+=1
            if c.max_steps and global_step>=c.max_steps: break
        v=metrics(model,val,device); row={"epoch":ep+1,"train/accuracy":correct/seen,"train/nll":loss_sum/seen,"val/accuracy":v["accuracy"],"val/nll":v["nll"],"val/ece":v["ece"],"objective":v["accuracy"],"lr":opt.param_groups[0]["lr"]}
        if device.type=="cuda": row["peak_training_bytes"]=torch.cuda.max_memory_allocated()
        row.update({f"model/{k}":v for k,v in stats.items()}); wb.log(row,step=ep+1); history.append(row); print(json.dumps(row),flush=True)
        if v["accuracy"]>best:
            best=v["accuracy"]; torch.save({"model":model.state_dict(),"config":asdict(c),"val":v},out/"best.pt")
        torch.save({"model":model.state_dict(),"config":asdict(c),"epoch":ep+1},out/"latest.pt")
        sched.step()
        if c.max_steps and global_step>=c.max_steps: break
    ck=torch.load(out/"best.pt",map_location=device,weights_only=False); model.load_state_dict(ck["model"])
    result={"config":asdict(c),"model":stats,"best_val":ck["val"],"profile_batch1":profile(model,device,1),"profile_batch128":profile(model,device,min(128,c.batch_size))}
    if test:
        result["test"]=metrics(model,test,device); wb.log({"eval/test_accuracy":result["test"]["accuracy"],"eval/test_nll":result["test"]["nll"],"eval/test_ece":result["test"]["ece"],"objective":result["test"]["accuracy"]})
    (out/"metrics.json").write_text(json.dumps(result,indent=2)); (out/"history.json").write_text(json.dumps(history,indent=2)); wb.finish(); print("RESULT="+json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser()
    for k,f in Config.__dataclass_fields__.items():
        flag="--"+k.replace("_","-")
        if isinstance(f.default,bool): p.add_argument(flag,action="store_true")
        else: p.add_argument(flag,type=type(f.default),default=f.default)
    a=vars(p.parse_args()); run(Config(**a))
if __name__=="__main__": main()
