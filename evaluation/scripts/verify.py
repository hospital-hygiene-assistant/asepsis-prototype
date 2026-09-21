"""Compact per-question verdict: the single best answer-bearing sentence,
searched first near the cited page, then across the whole document."""
import re, sys
sys.path.insert(0,'/private/tmp/claude-501/-Users-fede-Desktop-VSCODEProjects-asepsis-prototype/58cc9f68-d02b-407e-a87a-f167d41355eb/scratchpad')
from review import load
from find import keys, sentences

_cache={}
def doc(stem):
    if stem not in _cache: _cache[stem]=load(stem)
    return _cache[stem]

def best(stem, question, page, win):
    nums, words = keys(question)
    cands=[c for c in doc(stem) if c['pages'] and any(abs(p-page)<=win for p in c['pages'])] if win is not None else doc(stem)
    out=[]
    for c in cands:
        for s in sentences(c['body']):
            sl=s.lower()
            sc=sum(5 for n in nums if n.lower() in sl)+sum(1 for w in words if w.lower() in sl)
            if re.search(r'\d', s): sc+=2
            if sc>0: out.append((sc,c,s))
    out.sort(key=lambda x:-x[0])
    return out

def verdict(stem, qno, question, page, width=150):
    near=best(stem, question, page, win=1)
    if near and near[0][0]>=8:
        sc,c,s=near[0]
        return ('OK', qno, page, c['id'], sorted(c['pages'])[:5], sc, s[:width])
    wide=best(stem, question, page, win=None)
    if wide:
        sc,c,s=wide[0]
        pg=sorted(c['pages'])[:5]
        tag='OFFPAGE' if not any(abs(p-page)<=1 for p in c['pages']) else 'WEAK'
        return (tag, qno, page, c['id'], pg, sc, s[:width])
    return ('NONE', qno, page, '-', [], 0, '')
