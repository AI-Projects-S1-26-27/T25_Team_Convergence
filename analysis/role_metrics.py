import sys, json, numpy as np, collections
REPO=sys.argv[1]; sys.path.insert(0,REPO)
from experiments.evaluate import evaluate
from experiments.generate_scenarios import load_scenarios, TEST_PATH
from rl.agent import TeamPolicy
import rl.observation as of, rl.observation_no_teammate as on
from baselines.direct_pursuit import DirectPursuit
from baselines.predictive_interception import PredictiveInterception
from baselines.coordinated_pursuit import CoordinatedPursuit
scen=load_scenarios(TEST_PATH)
def m(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
def stats(res):
    stay=0; n=0; cap=collections.Counter(); far=[]
    for r in res:
        T=r['trajectory']
        for t in range(len(T)-1):
            P,q=T[t]; P2,_=T[t+1]
            stay+=sum(tuple(P[i])==tuple(P2[i]) for i in range(3)); n+=3
            far.append(max(m(p,q) for p in P))
        if r['captured']:
            P,q=T[-1]; Pp,qp=T[-2]
            for i in range(3):
                if tuple(P[i])==tuple(q) or (tuple(Pp[i])==tuple(q) and tuple(P[i])==tuple(qp)): cap[i]+=1; break
    tot=sum(cap.values())
    return dict(stay=stay/n, top_share=max(cap.values())/tot if tot else 0, far=float(np.mean(far)))
out={}
runs={'full':(of.encode_team_observations,16200),'ablation_no_teammate':(on.encode_team_observations,1080),'ablation_no_shaping':(of.encode_team_observations,16200)}
for run,(enc,ns) in runs.items():
    out[run]=[stats(evaluate(TeamPolicy.from_dir(f'{REPO}/results/{run}/best/seed_{s}',enc,ns),scen,record=True)) for s in range(10)]
for c in (DirectPursuit,PredictiveInterception,CoordinatedPursuit):
    out[c.name]=[stats(evaluate(c(seed=0),scen,record=True))]
summ={k:{kk:(float(np.mean([x[kk] for x in v])),float(np.min([x[kk] for x in v])),float(np.max([x[kk] for x in v]))) for kk in v[0]} for k,v in out.items()}
json.dump(dict(per_seed=out,summary=summ),open('out/roles.json','w'),indent=1)
for k,v in summ.items(): print(k,{a:tuple(round(x,3) for x in b) for a,b in v.items()})
