"""Print, per question, every chunk touching the cited page (+/- window) so the
answer location can be judged by reading rather than by term overlap."""
import csv, json, re, sys
from pathlib import Path
ROOT=Path('/Users/fede/Desktop/VSCODEProjects/asepsis-prototype')

def load(stem):
    lines=(ROOT/'knowledge_base'/f'{stem}.md').read_text().split('\n')
    idx=[i for i,l in enumerate(lines) if re.match(r'^#{1,6} ',l)]
    out=[]
    for k,i in enumerate(idx):
        title=re.match(r'^#{1,6} (.+)$',lines[i]).group(1).strip()
        pages=set(); nid=None
        j=i+1
        while j<len(lines) and not re.match(r'^#{1,6} ',lines[j]):
            m=re.match(r'^id: (.+)$',lines[j]);  nid = m.group(1) if m else nid
            m=re.match(r'^page: (\d+)$',lines[j])
            if m: pages.add(int(m.group(1)))
            m=re.match(r'^regions: (.+)$',lines[j])
            if m:
                try: pages.update(int(r[0]) for r in json.loads(m.group(1)))
                except Exception: pass
            j+=1
        end=idx[k+1] if k+1<len(idx) else len(lines)
        body=[l for l in lines[i+1:end] if l.strip() and not l.startswith('```')
              and not re.match(r'^(id|kind|doc|page|bbox|regions|scale|asset|type|image|golden):',l)]
        out.append({'id':nid,'title':title,'pages':pages,'body':' '.join(body)})
    return out

def show(stem, qno, question, page, win=1, chars=460):
    ch=load(stem)
    hits=[c for c in ch if c['pages'] and any(abs(p-page)<=win for p in c['pages'])]
    print(f"\n{'='*100}\nQ{qno}  [cited page {page}]  {question}")
    if not hits: print("   !! no chunk covers this page")
    for c in hits:
        print(f"  - {c['id'][:58]:60} pages={sorted(c['pages'])[:8]}")
        print(f"      {c['body'][:chars]}")
