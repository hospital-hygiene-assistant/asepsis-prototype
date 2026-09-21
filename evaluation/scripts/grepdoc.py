import re, sys
sys.path.insert(0,'/private/tmp/claude-501/-Users-fede-Desktop-VSCODEProjects-asepsis-prototype/58cc9f68-d02b-407e-a87a-f167d41355eb/scratchpad')
from review import load
from find import sentences
_c={}
def g(stem, pattern, label='', width=165, maxhits=3):
    if stem not in _c: _c[stem]=load(stem)
    rx=re.compile(pattern, re.I)
    hits=[]
    for c in _c[stem]:
        for s in sentences(c['body']):
            if rx.search(s): hits.append((c,s))
    print(f"  {label:10} /{pattern}/ -> {len(hits)} hit(s)")
    for c,s in hits[:maxhits]:
        print(f"     [{c['id'][:46]:48} p{sorted(c['pages'])[:4]}] {s[:width]}")
