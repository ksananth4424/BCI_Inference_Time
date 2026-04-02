from pathlib import Path
import re
import json
import pypdf

paper_dir = Path('papers')
out = []
for pdf in sorted(paper_dir.glob('*.pdf')):
    try:
        reader = pypdf.PdfReader(str(pdf))
        n = len(reader.pages)
        text = ''
        for i in range(min(12, n)):
            text += '\n' + (reader.pages[i].extract_text() or '')

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        title = ''
        for s in lines[:120]:
            if 20 < len(s) < 220 and not re.search(r'arXiv|Proceedings|copyright|accepted|preprint', s, re.I):
                title = s
                break

        m = re.search(r'(?is)abstract\s*(.{200,1400}?)\n\s*(?:1\s+introduction|introduction)', text)
        abstract = (m.group(1).strip().replace('\n', ' ') if m else '')

        benches = sorted(
            set(
                re.findall(
                    r'\b(POPE|MME|MMMU|MathVista|GQA|CHAIR|MMHal-Bench|MM-Vet|SEED-Bench|ScienceQA|OK-VQA|VQA v2|TextVQA|DocVQA|RefCOCO|CLEVR|AMBER|MMStar)\b',
                    text,
                    re.I,
                )
            )
        )
        metrics = sorted(
            set(
                re.findall(
                    r'\b(Accuracy|F1|Precision|Recall|CHAIRs|CHAIRi|CLIP-Score|CIDEr|IoU|Exact Match|EM|Pass@K|Hallucination)\b',
                    text,
                    re.I,
                )
            )
        )
        table_hits = len(re.findall(r'\bTable\b', text))

        out.append(
            {
                'file': pdf.name,
                'pages': n,
                'title': title,
                'benchmarks': benches,
                'metrics': metrics,
                'tables_first12': table_hits,
                'abstract_snippet': abstract[:900],
            }
        )
    except Exception as e:
        out.append({'file': pdf.name, 'error': str(e)})

out_path = Path('results/paper_review_index.json')
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2))
print('wrote', len(out), 'entries to', out_path)
for row in out:
    print('-', row['file'])
