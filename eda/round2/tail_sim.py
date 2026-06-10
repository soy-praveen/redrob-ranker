# Tail-strategy simulator: MAP/NDCG@50 tail policies, ghost placement, availability weighting.
# Reference date 2026-06-10. Pools from consensus paragraph grades (eda/round2/paragraph_grades_consensus.json).
import pandas as pd, numpy as np

rng0 = np.random.default_rng(7)
df = pd.read_pickle('eda/round2/_df_cache.pkl')
df = df.set_index('candidate_id', drop=False)
HP = set(pd.read_csv('eda/honeypot_candidates.csv').iloc[:,0])
for g in [5,4,3]:
    ov = df[(df['best_grade']==g) & df['candidate_id'].isin(HP)]['candidate_id'].tolist()
    print(f'honeypots in grade-{g} pool: {len(ov)} {ov}')
df = df[~df['candidate_id'].isin(HP)]               # hard-exclude all 93 honeypots

E   = df[df['best_grade']==5]                       # elite (T_ELITE + T_PLAIN), honeypots removed
G4  = df[df['best_grade']==4]                       # 139
G3  = df[df['best_grade']==3]                       # 963
SG3 = G3[G3['pids'].map(lambda P: ('P22' in P) or ('P30' in P))]  # 336 strong-G3 (round-1 grade-4 demotees)
G2  = df[df['best_grade']==2]                       # 5048 -> background tier 1
POOL = pd.concat([E, G4, G3])
GHOSTS6 = ['CAND_0007411','CAND_0033861','CAND_0041611','CAND_0060072','CAND_0092278','CAND_0094759']

# ---------- availability ----------
resp = df['sig_response_rate'].clip(lower=0)
act  = df['days_since_active']
noti = df['sig_notice_period_days']
resp_s = (resp/0.6).clip(0,1)
act_s  = (1-(act-30)/120).clip(0,1)
not_s  = pd.Series(np.select([noti<=30, noti<=60, noti<=90],[1.0,0.8,0.5],0.25), index=df.index)
A01 = 0.45*resp_s + 0.35*act_s + 0.20*not_s
HARD_GHOST = (resp<0.2)&(act>100)
SOFT_DEAD  = (resp<0.3)|(act>120)

# small engagement prior for tie-breaks (z-scored, clipped)
def z(s): s=s.astype(float); return ((s-s.mean())/(s.std()+1e-9)).clip(-2,2)
PRIOR = (z(df['sig_search_appearance_30d'])+z(df['sig_saved_by_recruiters_30d'])
         +z(df['sig_profile_views_30d'])+z(df['sig_profile_completeness']))/4  # ~[-2,2]

CBASE = df['best_grade'].map({5:100,4:80,3:60,2:30,1:10,0:0,-1:0}).astype(float)

def amult(amin):
    a = amin + (1.1-amin)*A01
    a[HARD_GHOST] = amin
    return a

# ---------- ground truth ----------
def make_gt(scen, mode, rng, noise=0.15, ghost_tier_override=None):
    """returns dict id->tier for POOL+G2; others tier 0."""
    t = pd.Series(0, index=df.index, dtype=float)
    t[G2.index]=1; t[G3.index]=2; t[G4.index]=4; t[E.index]=5
    if scen=='A180':
        pick = rng.choice(SG3.index, 12, replace=False); t[pick]=3
    elif scen=='B504':
        t[SG3.index]=3
    elif scen=='C1068':
        t[G3.index]=3
    if mode=='avail':
        gt2 = 2 if ghost_tier_override is None else ghost_tier_override
        t[HARD_GHOST & (t>=3)] = gt2
        sd = SOFT_DEAD & ~HARD_GHOST & (t>=3)
        t[sd] = t[sd]-1
    elif ghost_tier_override is not None:          # content mode w/ explicit ghost tier
        hg = df.index.isin(GHOSTS6)
        t[hg] = ghost_tier_override
    if noise>0:
        m = (t>=1)
        flips = rng.random(m.sum())<noise
        sgn = rng.choice([-1.,1.], m.sum())
        tv = t[m].to_numpy(); tv[flips]+=sgn[flips]; t[m]=np.clip(tv,0,5)
    return t

# ---------- metrics ----------
def composite(sub, t):
    g = t.reindex(sub).fillna(0).to_numpy()
    gains = 2**g - 1
    disc = 1/np.log2(np.arange(2, len(sub)+2))
    allt = np.sort(t.to_numpy())[::-1]
    ideal = 2**allt - 1
    def ndcg(k):
        dcg = (gains[:k]*disc[:k]).sum()
        idcg = (ideal[:k]*disc[:k]).sum()
        return dcg/idcg if idcg>0 else 0.0
    rel = (g>=3).astype(float)
    R = int((t.to_numpy()>=3).sum())
    cum = np.cumsum(rel)
    ap = (rel*(cum/np.arange(1,len(sub)+1))).sum()/max(min(R,len(sub)),1)
    p10 = rel[:10].mean()
    n10, n50 = ndcg(10), ndcg(50)
    return 0.5*n10+0.3*n50+0.15*ap+0.05*p10, n10, n50, ap, p10

# ---------- fixed head (ranks 1-39) ----------
S0 = CBASE*amult(0.5) + PRIOR
head_pool = pd.concat([E,G4])
head39 = head_pool.assign(S=S0[head_pool.index]).sort_values('S',ascending=False).index[:39].tolist()
rest = POOL.index.difference(head39)
restE4 = [i for i in rest if df.loc[i,'best_grade']>=4]
restG3 = [i for i in rest if df.loc[i,'best_grade']==3]
print(f"head39: elites={sum(df.loc[head39,'best_grade']==5)}, G4={sum(df.loc[head39,'best_grade']==4)}, "
      f"ghosts_in_head={sum(i in GHOSTS6 for i in head39)}")
print(f"rest grade>=4: {len(restE4)} (ghost elites among them: {sum(i in GHOSTS6 for i in restE4)}), G3: {len(restG3)}")

def order(ids, key):
    s = key.reindex(ids)
    return list(s.sort_values(ascending=False).index)

TAILS = {
 'a_G3_by_signals'  : order(restG3, A01*10+PRIOR)[:61],
 'b_G4_content_only': order(restE4, CBASE+0.01*PRIOR)[:61],
 'c_mixed_CxA'      : order(restE4+restG3, S0)[:61],
}
SUBS = {k: head39+v for k,v in TAILS.items()}
for k,v in SUBS.items():
    bg = df.loc[v[39:], 'best_grade']
    print(k, 'tail comp: g5=%d g4=%d g3=%d | ghosts in top100=%d' %
          ((bg==5).sum(),(bg==4).sum(),(bg==3).sum(), sum(i in GHOSTS6 for i in v)))

# ---------- Task 2: scenario x mode x policy ----------
print('\n=== TASK 2: tail policy x scenario (mean of 30 noise reps, noise=0.15; columns: composite | n10 n50 map p10) ===')
REPS=30
rows=[]
for scen in ['A180','B504','C1068']:
    for mode in ['avail','content']:
        res={}
        for pol,sub in SUBS.items():
            acc=np.zeros(5)
            for r in range(REPS):
                t = make_gt(scen, mode, np.random.default_rng(1000+r))
                acc += np.array(composite(sub,t))
            res[pol]=acc/REPS
        base = res['c_mixed_CxA'][0]
        line = f"{scen:6s} {mode:7s} | " + " | ".join(
            f"{p[:1]}: {v[0]:.4f} (d={v[0]-base:+.4f})" for p,v in res.items())
        print(line)
        for p,v in res.items():
            rows.append(dict(scen=scen,mode=mode,policy=p,comp=v[0],n10=v[1],n50=v[2],map=v[3],p10=v[4]))
pd.DataFrame(rows).to_csv('eda/round2/_task2_grid.csv', index=False)
g = pd.DataFrame(rows).groupby('policy')['comp']
print('\npolicy robustness over 6 cells: mean / min / worst-cell-regret')
piv = pd.DataFrame(rows).pivot_table(index=['scen','mode'],columns='policy',values='comp')
regret = (piv.max(axis=1).values[:,None]-piv).max(axis=0)
print(pd.DataFrame({'mean':g.mean(),'min':g.min(),'max_regret':regret}).round(4).to_string())

# ---------- Task 3: ghost placement ----------
print('\n=== TASK 3: 6 elite ghosts placement (policy c base; ghosts inserted as block) ===')
base = [i for i in SUBS['c_mixed_CxA'] if i not in GHOSTS6]
def insert_at(pos):           # 1-based start rank
    L = base[:pos-1]+GHOSTS6+base[pos-1:]
    return L[:100]
PLACE = {'ranks_11_16':insert_at(11),'ranks_31_36':insert_at(31),'ranks_45_50':insert_at(45),
         'ranks_75_80':insert_at(75),'ranks_95_100':insert_at(95),'out_of_100':base[:100]}
print(f"{'placement':14s} " + " ".join(f"{s}/{gt}".rjust(14) for s in ['A180','B504','C1068'] for gt in [2,4,5]))
res3={}
for nm,sub in PLACE.items():
    vals=[]
    for scen in ['A180','B504','C1068']:
        for gt in [2,4,5]:
            mode = 'avail' if gt==2 else 'content'
            acc=0
            for r in range(REPS):
                t = make_gt(scen, mode, np.random.default_rng(2000+r), ghost_tier_override=gt)
                acc += composite(sub,t)[0]
            vals.append(acc/REPS)
    res3[nm]=vals
    print(f"{nm:14s} " + " ".join(f"{v:14.4f}" for v in vals))
print('\nexpected composite under P(ghost-tier): mixtures of (t2,t4,t5) avg over scenarios')
for w,lbl in [((1,0,0),'100% t2'),((.6,.3,.1),'60/30/10'),((.5,.3,.2),'50/30/20'),((.33,.33,.34),'uniform'),((0,.5,.5),'content-only')]:
    print(lbl.rjust(14), {nm: round(np.mean([w[0]*v[i*3]+w[1]*v[i*3+1]+w[2]*v[i*3+2] for i in range(3)]),4) for nm,v in res3.items()})

# ---------- Task 4: multiplicative vs additive availability ----------
print('\n=== TASK 4: availability weighting (top-100 by score over the 1131 pool; 6 GT cells x 30 reps) ===')
def evaluate(score):
    sub = list(score.sort_values(ascending=False).index[:100])
    cells=[]
    for scen in ['A180','B504','C1068']:
        for mode in ['avail','content']:
            acc=0
            for r in range(REPS):
                acc += composite(sub, make_gt(scen,mode,np.random.default_rng(3000+r)))[0]
            cells.append(acc/REPS)
    return np.mean(cells), np.min(cells), cells
print('multiplicative S = C * (amin + (1.1-amin)*A01), ghosts floored at amin')
for amin in [0.4,0.5,0.6,0.75,0.9,1.05]:
    s = CBASE[POOL.index]*amult(amin)[POOL.index] + 0.3*PRIOR[POOL.index]
    m,mn,cells = evaluate(s)
    print(f"  amin={amin:4.2f}: mean={m:.4f} min={mn:.4f} cells={[round(c,3) for c in cells]}")
print('additive S = C + w*100*A01 (no ghost floor)')
for w in [0.0,0.05,0.1,0.2,0.3,0.5]:
    s = CBASE[POOL.index] + w*100*A01[POOL.index] + 0.3*PRIOR[POOL.index]
    m,mn,cells = evaluate(s)
    print(f"  w={w:4.2f}: mean={m:.4f} min={mn:.4f} cells={[round(c,3) for c in cells]}")
print('additive WITH ghost demotion term S = C + w*100*A01 - 25*ghost')
for w in [0.1,0.2,0.3]:
    s = CBASE[POOL.index] + w*100*A01[POOL.index] - 25*HARD_GHOST[POOL.index].astype(float) + 0.3*PRIOR[POOL.index]
    m,mn,cells = evaluate(s)
    print(f"  w={w:4.2f}: mean={m:.4f} min={mn:.4f}")

# ---------- Task 5: substructure within the 168 grade>=4 ----------
print('\n=== TASK 5: signal substructure within the 168 (E29+G4) ===')
P168 = pd.concat([E,G4])
X = P168[['sig_response_rate','days_since_active']].astype(float)
print(X.describe().round(3).to_string())
print('\nresponse_rate histogram (168):')
cnt,edges = np.histogram(X['sig_response_rate'], bins=np.arange(0,1.05,0.1))
for c,e in zip(cnt,edges): print(f"  {e:.1f}-{e+0.1:.1f}: {'#'*c} {c}")
print('days_since_active histogram (168):')
cnt,edges = np.histogram(X['days_since_active'], bins=np.arange(0,240,20))
for c,e in zip(cnt,edges): print(f"  {int(e):3d}-{int(e)+20:3d}: {'#'*c} {c}")
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
Xs = StandardScaler().fit_transform(X)
for k in [1,2,3]:
    gm = GaussianMixture(k, random_state=0, n_init=5).fit(Xs)
    print(f"GMM k={k}: BIC={gm.bic(Xs):.1f}")
km = KMeans(2, n_init=10, random_state=0).fit(Xs)
P168 = P168.assign(cl=km.labels_)
print('\nkmeans k=2 clusters (E29+G4):')
print(P168.groupby('cl').agg(n=('candidate_id','size'), resp=('sig_response_rate','median'),
      act=('days_since_active','median'), g5=('best_grade',lambda s:(s==5).sum()),
      notice=('sig_notice_period_days','median'), salmax=('sig_salary_max','median')).round(3).to_string())
# G4-only check
Xg = StandardScaler().fit_transform(G4[['sig_response_rate','days_since_active']])
for k in [1,2]:
    gm = GaussianMixture(k, random_state=0, n_init=5).fit(Xg)
    print(f"G4-only GMM k={k}: BIC={gm.bic(Xg):.1f}")
kmg = KMeans(2, n_init=10, random_state=0).fit(Xg)
g4c = G4.assign(cl=kmg.labels_)
print(g4c.groupby('cl').agg(n=('candidate_id','size'), resp=('sig_response_rate','median'),
      act=('days_since_active','median'), notice=('sig_notice_period_days','median')).round(3).to_string())
# does substructure align with paragraph ids?
g4c['toppid'] = g4c['pids'].map(lambda P: sorted(set(p for p in P if p in {'P27','P28','P29','P31'}))[0] if any(p in {'P27','P28','P29','P31'} for p in P) else '?')
print('\nG4 cluster x top grade-4 paragraph crosstab:')
print(pd.crosstab(g4c['cl'], g4c['toppid']).to_string())
print('\nE29 signals (sorted by resp):')
print(E[['candidate_id','sig_response_rate','days_since_active','sig_notice_period_days','sig_salary_max']]
      .sort_values('sig_response_rate').to_string(index=False))
