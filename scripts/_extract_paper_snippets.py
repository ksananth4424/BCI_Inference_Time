from pathlib import Path
import re
import json
import pypdf

KEYS = [
    'abstract',
    'introduction',
    'method',
    'approach',
    'implementation',
    'experimental setup',
    'results',
    'table',
    'benchmark',
    'decoder',
    'verification',
    'hallucination',
]

paper_dir = Path('papers')
all_notes = []

for pdf in sorted(paper_dir.glob('*.pdf')):
    reader = pypdf.PdfReader(str(pdf))
    text = ''
    for i in range(len(reader.pages)):
        t = reader.pages[i].extract_text() or ''
        text += f"\n\n[PAGE {i+1}]\n" + t

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = ''
    for s in lines[:140]:
        if 20 < len(s) < 220 and not re.search(r'arXiv|Proceedings|copyright|preprint|openaccess', s, re.I):
            title = s
            break

    snippets = {}
    lower_text = text.lower()
    for k in KEYS:
        idx = lower_text.find(k)
        if idx >= 0:
            start = max(0, idx - 200)
            end = min(len(text), idx + 900)
            snippets[k] = text[start:end].replace('\n', ' ')[:900]

    # Collect table mentions with page refs
    table_pages = sorted(set(int(m.group(1)) for m in re.finditer(r'\[PAGE\s+(\d+)\][\s\S]{0,1200}?\bTable\b', text, re.I)))

    benches = sorted(set(re.findall(r'\b(POPE|MME|MMMU|MMStar|MathVista|GQA|CHAIR|MMHal-Bench|MM-Vet|SEED-Bench|ScienceQA|OK-VQA|VQA v2|TextVQA|DocVQA|RefCOCO|CLEVR|AMBER|GAVIE)\b', text, re.I)))

    all_notes.append({
        'file': pdf.name,
        'title': title,
        'pages': len(reader.pages),
        'table_pages': table_pages,
        'benchmarks': benches,
        'snippets': snippets,
    })

out = Path('results/paper_snippets.json')
out.write_text(json.dumps(all_notes, indent=2))
print('wrote', out)
