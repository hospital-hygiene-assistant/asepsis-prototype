"""For a question, show the SENTENCES in each candidate chunk that carry its
distinctive terms — so the answer can be confirmed by reading it."""
import re, sys
sys.path.insert(0,'/private/tmp/claude-501/-Users-fede-Desktop-VSCODEProjects-asepsis-prototype/58cc9f68-d02b-407e-a87a-f167d41355eb/scratchpad')
from review import load

STOP=set('the a an of to in for and or is are was were be been what which who how many much '
 'according report guidelines study cited does do did at on by with from that this their its '
 'it as per into over under about were used when where whose any one two all not no more most '
 'other others such than then there these those also only into within across during'.split())

def keys(q):
    nums=[n for n in re.findall(r'\b\d[\d.,]*\b', q)]
    words=[w for w in re.findall(r"\b[A-Za-z][A-Za-z\-']{2,}\b", q) if w.lower() not in STOP]
    return nums, words

def sentences(txt):
    return [s.strip() for s in re.split(r'(?<=[.;:])\s+', txt) if s.strip()]

def find(stem, qno, question, page, win=1, per=3, width=230):
    nums, words = keys(question)
    ch=[c for c in load(stem) if c['pages'] and any(abs(p-page)<=win for p in c['pages'])]
    print(f"\n{'='*98}\nQ{qno} [p{page}] {question}")
    for c in ch:
        best=[]
        for s in sentences(c['body']):
            sl=s.lower()
            sc=sum(4 for n in nums if n.lower() in sl)+sum(1 for w in words if w.lower() in sl)
            # a sentence that answers a "how many/what percentage" question needs a number
            if re.search(r'\d', s): sc+=2
            if sc: best.append((sc,s))
        best.sort(key=lambda x:-x[0])
        if not best: continue
        print(f"  [{c['id'][:52]}]  pages={sorted(c['pages'])[:6]}")
        for sc,s in best[:per]:
            print(f"      ({sc:>2}) {s[:width]}")
