"""Set B (fictional-term queries, full pipeline) and Set C (forced pairing).

Everything is logged per query: the passages handed to the model, the answer,
the completeness verdict, and whether the invented term was named.
"""
import sys, csv, json, time, random, argparse
sys.path.insert(0,'/Users/fede/Desktop/VSCODEProjects/asepsis-prototype')
sys.path.insert(0,'/private/tmp/claude-501/-Users-fede-Desktop-VSCODEProjects-asepsis-prototype/58cc9f68-d02b-407e-a87a-f167d41355eb/scratchpad')
from goldrule import score as gold_score, doc_info
import synthesis
from modules.index import pageindex_custom as ix
import pageindex as _pi
from pathlib import Path

ROOT=Path('/Users/fede/Desktop/VSCODEProjects/asepsis-prototype')
OUT=Path('/private/tmp/claude-501/-Users-fede-Desktop-VSCODEProjects-asepsis-prototype/58cc9f68-d02b-407e-a87a-f167d41355eb/scratchpad')

def save(name, obj): json.dump(obj, open(OUT/name,'w'), indent=1, ensure_ascii=False)

def all_leaves():
    """Every leaf in the corpus, with its golden labels (if any)."""
    out=[]
    for f in sorted((ROOT/'index').glob('*.json')):
        d=json.load(open(f))
        def w(ns, crumb=''):
            for n in ns:
                path=f"{crumb} > {n['title']}" if crumb else n['title']
                kids=n.get('children') or []
                if not kids and (n.get('content') or '').strip():
                    out.append({'doc':f.stem,'node_id':n['nodeId'],'title':n['title'],
                                'breadcrumb':path,'excerpt':n['content'],
                                'golden':(n.get('pin') or {}).get('golden')})
                w(kids, path)
        w(d['nodes'])
    return out

def run_B(limit=None):
    rows=list(csv.DictReader(open(ROOT/'evaluation/queries/asepsis_100_fictional.csv')))[:limit]
    try:    recs=json.load(open(OUT/'setB.json'))
    except Exception: recs=[]
    done={x['id'] for x in recs}
    if done: print(f"resuming — {len(done)} already done", flush=True)
    t0=time.time()
    for k,r in enumerate(rows,1):
        if r['id'] in done: continue
        t=time.time()
        nodes, meta = ix.retrieve_with_metadata(r['doc'], r['question'])
        got=[n.node_id for n in nodes]
        hit, tgts, matched = gold_score(r['doc'], r['node_id'], got)
        srcs=synthesis.number_sources([{'doc':r['doc'],'node_id':n.node_id,'title':n.title,
                                        'breadcrumb':_pi._make_breadcrumb(n.node_id,{},{}) or n.title,
                                        'excerpt':n.content or ''} for n in nodes])
        try:
            res=synthesis.synthesise(r['question'], srcs, completeness=True)
            ans, judg = res.answer, res.judgment
        except Exception as e:
            ans, judg = f'ERROR: {e}', {'label':'error','still_needed':[]}
        term=r['edit_term']; sn=' '.join(judg.get('still_needed') or [])
        rec={'id':r['id'],'pair_id':r['pair_id'],'doc':r['doc'],'question':r['question'],
             'edit_term':term,'gold_node':r['node_id'],
             'retrieved':got,'satisfying_targets':tgts,'matched':matched,
             'gold_retrieved':hit,
             'gold_status':[ (meta.get(x) or {}).get('status','never a candidate') for x in tgts],
             'gold_rehydrated':any((meta.get(x) or {}).get('rehydrated') for x in tgts),
             'n_sources':len(srcs),
             'judgment_label':judg.get('label'),'still_needed':judg.get('still_needed'),
             'names_edit_term_in_still_needed': term.lower() in sn.lower() if term else None,
             'names_edit_term_in_answer': term.lower() in (ans or '').lower() if term else None,
             'answer':ans,'secs':round(time.time()-t,1)}
        recs.append(rec); save('setB.json', recs)
        print(f"[{k:>3}/{len(rows)} {time.time()-t0:6.0f}s] {r['id']} judg={str(judg.get('label')):<12} "
              f"names_term={rec['names_edit_term_in_still_needed']} gold={'Y' if rec['gold_retrieved'] else 'n'} "
              f"{rec['secs']:5.0f}s", flush=True)
    ins=sum(1 for x in recs if x['judgment_label']=='insufficient')
    nam=sum(1 for x in recs if x['names_edit_term_in_still_needed'])
    print(f"\nSET B: insufficient {ins}/{len(recs)} | named the invented term {nam}/{len(recs)}", flush=True)

def run_C(k_chunks=5, seed=20260921, limit=None):
    rows=list(csv.DictReader(open(ROOT/'evaluation/queries/asepsis_100.csv')))[:limit]
    pool=[l for l in all_leaves() if not l.get('golden')]
    print(f"decoy pool: {len(pool)} leaves carrying no golden label", flush=True)
    rng=random.Random(seed)
    try:    recs=json.load(open(OUT/'setC.json'))
    except Exception: recs=[]
    done={x['id'] for x in recs}
    if done: print(f"resuming — {len(done)} already done", flush=True)
    t0=time.time()
    for k,r in enumerate(rows,1):
        # draw for every row so the seeded sequence is identical on a resume
        picked_pre=rng.sample(pool, k_chunks)
        if r['id'].replace('A','C') in done: continue
        t=time.time()
        picked=picked_pre
        srcs=synthesis.number_sources([{x:p[x] for x in ('doc','node_id','title','breadcrumb','excerpt')} for p in picked])
        try:
            res=synthesis.synthesise(r['question'], srcs, completeness=True)
            ans, judg = res.answer, res.judgment
        except Exception as e:
            ans, judg = f'ERROR: {e}', {'label':'error','still_needed':[]}
        rec={'id':r['id'].replace('A','C'),'pair_id':r['id'],'question':r['question'],
             'gold_node':r['node_id'],
             'decoys':[{'doc':p['doc'],'node_id':p['node_id'],'title':p['title']} for p in picked],
             'judgment_label':judg.get('label'),'still_needed':judg.get('still_needed'),
             'passed':judg.get('label')=='insufficient','answer':ans,'secs':round(time.time()-t,1)}
        recs.append(rec); save('setC.json', recs)
        print(f"[{k:>3}/{len(rows)} {time.time()-t0:6.0f}s] {rec['id']} label={str(judg.get('label')):<12} "
              f"{'PASS' if rec['passed'] else 'FAIL'} {rec['secs']:5.0f}s", flush=True)
    p=sum(1 for x in recs if x['passed'])
    print(f"\nSET C: said insufficient {p}/{len(recs)} ({p/max(len(recs),1):.0%})", flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('which'); ap.add_argument('--limit',type=int)
    a=ap.parse_args()
    (run_B if a.which=='B' else run_C)(limit=a.limit)
