import torch
from srn.models import PairwiseLUT, RankLUT, build_model, model_stats

def test_pairwise_deterministic_and_gradient():
    a=PairwiseLUT(12,5,3,4,seed=7); b=PairwiseLUT(12,5,3,4,seed=7)
    assert torch.equal(a.anchors_a,b.anchors_a)
    x=torch.randn(2,12,requires_grad=True); a(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert a.table.grad is not None

def test_rank_gradient():
    m=RankLUT(12,5,3,4,64,seed=3); x=torch.randn(2,12,requires_grad=True)
    m(x).sum().backward(); assert x.grad is not None and x.grad.abs().sum()>0

def test_all_model_shapes():
    x=torch.randn(2,3,32,32)
    for name in ("ann","pairwise","rank","lif"):
        m=build_model(name,width=4,blocks=1,tables=2,bits=3)
        assert m(x).shape==(2,10)
        assert model_stats(m)["resident_model_bytes"]>0
