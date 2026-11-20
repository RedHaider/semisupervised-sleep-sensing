import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def dataset(n=2600,seed=27):
 r=np.random.default_rng(seed);y=r.binomial(1,.55,n);x=r.normal(size=(n,7));x[:,0]+=y*1.1;x[:,1]-=y*.8;x[:,2]+=y*.6;x[:,3]+=y*x[:,4]*.45;return x,y
def evaluate(seed=27):
 x,y=dataset(seed=seed);lab=250;unlab=1700;test=x[unlab:];yt=y[unlab:];base=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500)).fit(x[:lab],y[:lab]);pu=base.predict_proba(x[lab:unlab])[:,1];keep=(pu<.08)|(pu>.92);pseudo=(pu[keep]>.5).astype(int);semi=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500)).fit(np.vstack([x[:lab],x[lab:unlab][keep]]),np.r_[y[:lab],pseudo]);out={}
 for name,m in {"supervised":base,"self_training":semi}.items():
  p=m.predict_proba(test)[:,1];out[name]={"auroc":float(roc_auc_score(yt,p)),"f1":float(f1_score(yt,p>.5))}
 out["pseudo_label_coverage"]=float(keep.mean());return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
