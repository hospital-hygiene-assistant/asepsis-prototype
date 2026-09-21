"""BM25 rank of each query's golden chunk among ALL leaves of its document.

No LLM. Uses the same ranker, tokeniser and text the system itself ranks with
(ranking.rank over title + summary + content), and the same gold rule as the
retrieval evals, so the two numbers are directly comparable.
"""
import sys, csv, json
sys.path.insert(0,'/Users/fede/Desktop/VSCODEProjects/asepsis-prototype')
sys.path.insert(0,'/private/tmp/claude-501/-Users-fede-Desktop-VSCODEProjects-asepsis-prototype/58cc9f68-d02b-407e-a87a-f167d41355eb/scratchpad')
import ranking
from goldrule import targets
from pathlib import Path
ROOT=Path('/Users/fede/Desktop/VSCODEProjects/asepsis-prototype')
OUT=Path('/private/tmp/claude-501/-Users-fede-Desktop-VSCODEProjects-asepsis-prototype/58cc9f68-d02b-407e-a87a-f167d41355eb/scratchpad')

_leaves={}
def leaves(stem):
    if stem in _leaves: return _leaves[stem]
    d=json.load(open(ROOT/'index'/f'{stem}.json')); out=[]
    def w(ns):
        for n in ns:
            kids=n.get('children') or []
            if not kids:
                out.append({'node_id':n['nodeId'],'title':n['title'],
                            'summary':n.get('summary') or '','content':n.get('content') or ''})
            w(kids)
    w(d['nodes']); _leaves[stem]=out; return out

rows=list(csv.DictReader(open(ROOT/'evaluation/queries/asepsis_100.csv')))
recs=[]
for r in rows:
    L=leaves(r['doc'])
    tg=set(targets(r['doc'], r['node_id']))
    ranked=ranking.rank(L, r['question'],
                        text_of=lambda l: f"{l['title']}\n{l['summary']}\n{l['content']}")
    pos=None; sc=None
    for s in ranked:
        if s.item['node_id'] in tg: pos, sc = s.rank, s.score; break
    recs.append({'id':r['id'],'doc':r['doc'],'gold_node':r['node_id'],
                 'satisfying_targets':sorted(tg),'n_leaves':len(L),
                 'bm25_rank_0based':pos,'bm25_rank':None if pos is None else pos+1,
                 'bm25_score':None if sc is None else round(sc,4),
                 'top1_node':ranked[0].item['node_id'] if ranked else None,
                 'top1_score':round(ranked[0].score,4) if ranked else None})
json.dump(recs, open(OUT/'bm25_ranks.json','w'), indent=1, ensure_ascii=False)
found=[x for x in recs if x['bm25_rank']]
import statistics
print(f"golden chunk located by BM25: {len(found)}/{len(recs)}")
for k in (1,2,3,5,10,20,50):
    print(f"  recall@{k:<3} {sum(1 for x in found if x['bm25_rank']<=k):>3}/100")
print(f"  median rank {statistics.median(x['bm25_rank'] for x in found):.0f}  "
      f"mean {statistics.mean(x['bm25_rank'] for x in found):.1f}  worst {max(x['bm25_rank'] for x in found)}")
print("\nworst 8:")
for x in sorted(found, key=lambda x:-x['bm25_rank'])[:8]:
    print(f"  {x['id']} rank {x['bm25_rank']:>4}/{x['n_leaves']:<4} {x['satisfying_targets'][0][:50]}")
