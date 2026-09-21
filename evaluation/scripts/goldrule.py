"""How a gold label is satisfied.

Retrieval returns LEAVES. A gold label may name a leaf or a section:

  * leaf gold    -> satisfied only by that leaf.
  * section gold -> satisfied only by that section's SYNTHETIC OVERVIEW leaf,
    which carries the section's own prose. That prose is the text the label was
    verified against, so it is the only child that can contain the answer.

Explicitly NOT satisfied by any other descendant. A section's other children
are different passages — subsections, figure leaves, icon groups — and
accepting them would count retrieving a neighbouring figure as finding the
answer.
"""
import json
from pathlib import Path
ROOT=Path('/Users/fede/Desktop/VSCODEProjects/asepsis-prototype')
_cache={}

def doc_info(stem):
    if stem in _cache: return _cache[stem]
    d=json.load(open(ROOT/'index'/f'{stem}.json')); info={}
    def w(ns):
        for n in ns:
            kids=n.get('children') or []
            ov=next((c['nodeId'] for c in kids if c.get('synthetic')), None)
            info[n['nodeId']]={'leaf':not kids,'overview':ov,
                               'kids':[c['nodeId'] for c in kids],
                               'content_len':len(n.get('content') or '')}
            w(kids)
    w(d['nodes']); _cache[stem]=info; return info

def targets(stem, gold):
    """The node id(s) whose retrieval satisfies this gold label."""
    i=doc_info(stem).get(gold)
    if i is None: return [gold]
    if i['leaf']: return [gold]
    return [i['overview']] if i['overview'] else []

def score(stem, gold, retrieved_ids):
    t=targets(stem, gold)
    hit=[x for x in t if x in retrieved_ids]
    return bool(hit), t, hit
