import sys, csv, json, time
sys.path.insert(0,'/Users/fede/Desktop/VSCODEProjects/asepsis-prototype')
sys.path.insert(0,'/private/tmp/claude-501/-Users-fede-Desktop-VSCODEProjects-asepsis-prototype/58cc9f68-d02b-407e-a87a-f167d41355eb/scratchpad')
from modules.index import pageindex_custom as ix
from goldrule import score, doc_info
OUT='/private/tmp/claude-501/-Users-fede-Desktop-VSCODEProjects-asepsis-prototype/58cc9f68-d02b-407e-a87a-f167d41355eb/scratchpad/setA.json'
rows=list(csv.DictReader(open('evaluation/queries/asepsis_100.csv')))
# Resumable: records already written are kept and their queries skipped, so a
# stop costs at most the query in flight rather than the whole run.
try:    res=json.load(open(OUT))
except Exception: res=[]
done={x['id'] for x in res}
if done: print(f"resuming — {len(done)} already done, skipping them", flush=True)
t0=time.time()
for k,r in enumerate(rows,1):
    if r['id'] in done: continue
    t=time.time()
    nodes, meta = ix.retrieve_with_metadata(r['doc'], r['question'])
    got=[n.node_id for n in nodes]
    hit, tgts, matched = score(r['doc'], r['node_id'], got)
    gi=doc_info(r['doc']).get(r['node_id'], {})
    tm=[(meta.get(x) or {}) for x in tgts]
    res.append({'id':r['id'],'doc':r['doc'],'question':r['question'],
        'gold_node':r['node_id'],'gold_is_leaf':gi.get('leaf'),
        'satisfying_targets':tgts,'matched':matched,'hit':hit,
        'how':('rehydrated' if any(m.get('rehydrated') for m in tm) else ('walk' if hit else '-')),
        'target_status':[m.get('status','never a candidate') for m in tm],
        'target_reason':[str(m.get('reason',''))[:200] for m in tm],
        'n_retrieved':len(got),'retrieved':got,'secs':round(time.time()-t,1)})
    json.dump(res, open(OUT,'w'), indent=1, ensure_ascii=False)
    print(f"[{k:>3}/100 {time.time()-t0:6.0f}s] {r['id']} {'HIT ' if hit else 'MISS'} "
          f"{res[-1]['how']:11} n={len(got):<3} {res[-1]['secs']:5.0f}s  {tgts[0][:44] if tgts else '-'}", flush=True)
h=sum(1 for x in res if x['hit'])
print(f"\nSET A golden recall: {h}/{len(res)} ({h/len(res):.0%}) in {time.time()-t0:.0f}s", flush=True)
