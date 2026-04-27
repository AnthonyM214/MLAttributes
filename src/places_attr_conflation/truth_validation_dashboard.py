"""Conflict-first truth validation dashboard for Project A labels.

This module is intentionally standalone from the benchmark dashboard. It turns a
reviewset CSV into a fast local labeling UI that can export the golden-label CSV
schema used by golden.py.
"""

from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_A_ATTRIBUTES = ("website", "phone", "address", "category", "name")
TRUTH_CHOICES = ("current", "base", "both", "neither", "not_enough_evidence")


@dataclass(frozen=True)
class TruthDashboardData:
    review_rows: list[dict[str, str]]
    source_csv: str


def load_review_rows(path: str | Path, *, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open(newline="", encoding="utf-8", errors="replace") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if limit is not None and index >= limit:
                break
            rows.append(dict(row))
    return rows


def _escape(value: object) -> str:
    return html.escape(str(value if value is not None else ""))


def _json_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).replace("<", "\\u003c")


def _review_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    conflicts_by_attribute = {attribute: 0 for attribute in PROJECT_A_ATTRIBUTES}
    total_conflicts = 0
    for row in rows:
        for attribute in PROJECT_A_ATTRIBUTES:
            differs = str(row.get(f"{attribute}_differs", "")).lower() in {"true", "1", "yes"}
            if differs:
                conflicts_by_attribute[attribute] += 1
                total_conflicts += 1
    return {
        "rows": len(rows),
        "conflicts": total_conflicts,
        "conflicts_by_attribute": conflicts_by_attribute,
    }


def render_truth_validation_html(data: TruthDashboardData) -> str:
    summary = _review_summary(data.review_rows)
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            "<title>MLAttributes Truth Validation</title>",
            "<style>",
            ":root{--paper:#f4f1ea;--ink:#17201d;--muted:#5f685f;--accent:#1f5c4d;--blue:#174b68;--line:#d7cfbf;--panel:#fffdf8;--warn:#8a3b1e;}",
            "body{margin:0;background:linear-gradient(180deg,#ede7da 0%,var(--paper) 220px);color:var(--ink);font-family:Georgia,'Times New Roman',serif;}",
            "main{max-width:1180px;margin:0 auto;padding:34px 20px 70px;}h1,h2,h3,button,input,textarea,label{font-family:'Trebuchet MS','Gill Sans',sans-serif;}h1{margin:0;font-size:2.35rem}.lead{color:var(--muted);max-width:78ch}.panel{background:var(--panel);border:1px solid var(--line);padding:16px;margin:14px 0}.top{display:grid;grid-template-columns:1.7fr 1fr;gap:16px;align-items:start}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}.stat{background:#f7f2e8;border:1px solid var(--line);padding:12px;text-align:center}.stat b{display:block;color:var(--accent);font-size:1.55rem}.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0}.toolbar button,.choice{border:1px solid var(--line);background:#efe8d9;color:var(--ink);padding:9px 12px;border-radius:3px;cursor:pointer}.toolbar .primary,button.primary,.choice.selected{background:var(--accent);border-color:var(--accent);color:#fff}.choice-row{display:flex;flex-wrap:wrap;gap:7px;margin:10px 0}.queue{display:grid;grid-template-columns:1.05fr .95fr;gap:14px}.value-boxes{display:grid;grid-template-columns:1fr 1fr;gap:10px}.value-box{background:#f6efe2;border:1px solid var(--line);padding:10px;min-height:92px;word-break:break-word}.value-box h3{margin:0 0 6px}.conflict-badge{color:#fff;background:var(--warn);padding:2px 7px;border-radius:10px;font-size:.75rem;text-transform:uppercase}.field{display:block;margin:10px 0;color:var(--muted)}input,textarea{box-sizing:border-box;width:100%;border:1px solid var(--line);padding:8px;background:#fff;margin-top:4px}textarea{min-height:70px}.warn{color:var(--warn)}.hint{font-size:.92rem;color:var(--muted)}table{width:100%;border-collapse:collapse;background:var(--panel);margin-top:10px}th,td{border-bottom:1px solid var(--line);padding:9px;text-align:left;vertical-align:top}th{background:#d8e8de}.search-links a{display:inline-block;margin:3px 6px 3px 0}.hidden{display:none}@media(max-width:900px){.top,.queue,.value-boxes{grid-template-columns:1fr}h1{font-size:2rem}}",
            "</style>",
            "</head>",
            "<body>",
            f"<script id='review-data' type='application/json'>{_json_payload(data.review_rows)}</script>",
            f"<script id='summary-data' type='application/json'>{_json_payload(summary)}</script>",
            "<main>",
            "<section class='top'><div><h1>Truth Validation Queue</h1><p class='lead'>Conflict-first reviewer UI for Project A. Label only the attributes that need human judgment, export clean truth labels, and compare reviewers before labels become evaluation truth.</p></div><div class='panel'><b>Shortcuts</b><p class='hint'>1=current, 2=base, 3=both, 4=neither, 5=not enough evidence, N=next, B=back, E=evidence URL.</p></div></section>",
            "<section class='stats'><div class='stat'><b id='rowCount'>0</b><span>rows</span></div><div class='stat'><b id='conflictCount'>0</b><span>conflicts</span></div><div class='stat'><b id='decisionCount'>0</b><span>decisions</span></div><div class='stat'><b id='completePct'>0%</b><span>complete</span></div><div class='stat'><b id='uncertainCount'>0</b><span>uncertain</span></div><div class='stat'><b id='disagreementCount'>0</b><span>reviewer disagreements</span></div></section>",
            "<section class='panel'><div class='toolbar'><label>Reviewer <input id='reviewerName' value='reviewer_1'></label><label><input id='showAllToggle' type='checkbox'> Show non-conflicting attributes</label><button id='prevItem'>Back</button><button id='nextItem'>Next</button><button class='primary' id='exportLabels'>Export label CSV</button><button id='exportFollowup'>Export follow-up queue</button><button id='clearState'>Clear saved review</button></div><div class='toolbar'><label>Load review CSV <input id='reviewCsvInput' type='file' accept='.csv,text/csv'></label><label>Compare reviewer CSVs <input id='multiCsvInput' type='file' accept='.csv,text/csv' multiple></label></div><p id='position' class='hint'>No item loaded.</p><p id='qualityWarnings' class='warn'></p></section>",
            "<section class='queue'><div class='panel'><h2 id='itemTitle'>No conflict loaded</h2><p id='itemMeta' class='hint'></p><div class='value-boxes'><div class='value-box'><h3>Current</h3><p id='currentValue'>—</p></div><div class='value-box'><h3>Base</h3><p id='baseValue'>—</p></div></div><div class='search-links' id='searchLinks'></div><div class='choice-row' id='choiceButtons'></div></div><div class='panel'><h2>Evidence + notes</h2><label class='field'>Override truth value<input id='truthValue'></label><label class='field'>Evidence URL<input id='evidenceUrl'></label><label class='field'>Review notes<textarea id='reviewNotes'></textarea></label><p class='hint'>Use neither when both values are wrong, and fill the correct value manually. Use not enough evidence when it cannot be verified.</p></div></section>",
            "<section class='panel'><h2>Reviewer agreement by attribute</h2><div id='agreementSummary'></div><h2>Multi-reviewer disagreements</h2><div id='disagreementTable'><p class='hint'>Upload reviewer CSVs to compare labels.</p></div></section>",
            "</main>",
            "<script>",
            "const ATTRIBUTES=['website','phone','address','category','name']; const CHOICES=['current','base','both','neither','not_enough_evidence'];",
            "let reviewRows=JSON.parse(document.getElementById('review-data').textContent||'[]'); let pointer=0; let queue=[];",
            "function reviewerName(){return (document.getElementById('reviewerName').value||'reviewer_1').trim()||'reviewer_1'} function stateKey(){return `mlattrs_truth:${reviewerName()}`} function loadState(){try{return JSON.parse(localStorage.getItem(stateKey())||'{}')}catch{return {}}} function saveState(s){localStorage.setItem(stateKey(),JSON.stringify(s))} function rowKey(r){return `${r.id||''}|${r.base_id||''}`} function itemKey(item){return `${rowKey(item.row)}|${item.attribute}`} function esc(v){return String(v??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))} function csvEsc(v){const t=String(v??''); return /[\",\n]/.test(t)?'\"'+t.replaceAll('"','""')+'\"':t}",
            "function isConflict(row,a){return ['true','1','yes'].includes(String(row[`${a}_differs`]||'').toLowerCase()) || ((row[a]||'') && (row[`base_${a}`]||'') && row[a]!==row[`base_${a}`])}",
            "function rebuildQueue(){const showAll=document.getElementById('showAllToggle').checked; queue=[]; for(const row of reviewRows){for(const a of ATTRIBUTES){if(showAll||isConflict(row,a)) queue.push({row,attribute:a})}} if(pointer>=queue.length) pointer=Math.max(0,queue.length-1);}",
            "function decision(item){return loadState()[itemKey(item)]||{}} function currentItem(){return queue[pointer]}",
            "function saveCurrent(choice=null){const item=currentItem(); if(!item)return; const s=loadState(); const key=itemKey(item); const prior=s[key]||{}; s[key]={...prior, choice:choice||prior.choice||'', truth_value:document.getElementById('truthValue').value, evidence_url:document.getElementById('evidenceUrl').value, notes:document.getElementById('reviewNotes').value, reviewer:reviewerName(), updated_at:new Date().toISOString()}; saveState(s)}",
            "function choose(choice){saveCurrent(choice); const item=currentItem(); const nextUnlabeled=queue.findIndex((q,i)=>i>pointer && !decision(q).choice); if(nextUnlabeled>=0) pointer=nextUnlabeled; else pointer=Math.min(pointer+1,queue.length-1); render()}",
            "function render(){rebuildQueue(); const state=loadState(); let decisions=0, uncertain=0; for(const item of queue){const d=state[itemKey(item)]||{}; if(d.choice) decisions++; if(d.choice==='not_enough_evidence') uncertain++} document.getElementById('rowCount').textContent=reviewRows.length; document.getElementById('conflictCount').textContent=queue.length; document.getElementById('decisionCount').textContent=decisions; document.getElementById('uncertainCount').textContent=uncertain; document.getElementById('completePct').textContent=queue.length?`${Math.round(decisions/queue.length*100)}%`:'0%'; document.getElementById('position').textContent=queue.length?`Conflict ${pointer+1} of ${queue.length}`:'No conflicts loaded.'; const item=currentItem(); if(!item){document.getElementById('itemTitle').textContent='No conflict loaded'; return} const row=item.row, a=item.attribute, d=decision(item); document.getElementById('itemTitle').innerHTML=`${esc(a)} ${isConflict(row,a)?'<span class=conflict-badge>conflict</span>':''}`; document.getElementById('itemMeta').innerHTML=`id: ${esc(row.id||'-')} · base_id: ${esc(row.base_id||'-')}`; document.getElementById('currentValue').textContent=row[a]||'—'; document.getElementById('baseValue').textContent=row[`base_${a}`]||'—'; document.getElementById('truthValue').value=d.truth_value||''; document.getElementById('evidenceUrl').value=d.evidence_url||''; document.getElementById('reviewNotes').value=d.notes||''; document.getElementById('choiceButtons').innerHTML=CHOICES.map((c,i)=>`<button class='choice ${d.choice===c?'selected':''}' onclick=choose('${c}')>${i+1}. ${c.replaceAll('_',' ')}</button>`).join(''); document.getElementById('searchLinks').innerHTML=searchLinks(row,a); document.getElementById('qualityWarnings').textContent=qualityWarnings(state).join(' | '); renderAgreement([])}",
            "function searchLinks(row,a){const name=row.name||row.base_name||''; const addr=row.address||row.base_address||''; const vals=[row[a],row[`base_${a}`],`${name} ${addr}`].filter(Boolean); return vals.map((v,i)=>`<a target='_blank' rel='noreferrer' href='https://www.google.com/search?q=${encodeURIComponent(v)}'>search ${i+1}</a>`).join('')}",
            "function qualityWarnings(state){const w=[]; for(const item of queue){const d=state[itemKey(item)]||{}; const row=item.row, a=item.attribute; if(d.choice==='current'&&!row[a])w.push(`${a}: current chosen but blank`); if(d.choice==='base'&&!row[`base_${a}`])w.push(`${a}: base chosen but blank`); if(d.choice==='neither'&&!d.truth_value)w.push(`${a}: neither needs truth value`); if((d.choice==='current'||d.choice==='base'||d.choice==='both'||d.choice==='neither')&&!d.evidence_url)w.push(`${a}: missing evidence URL`)} return w.slice(0,4)}",
            "function headers(){return ['id','base_id','label_status','notes',...ATTRIBUTES.flatMap(a=>[`${a}_truth_choice`,`${a}_truth_value`,`${a}_evidence_url`,`${a}_label_source`])]} function exportRows(filterFn=null){const h=headers(), s=loadState(); const rows=reviewRows.map(row=>{const out={id:row.id||'',base_id:row.base_id||'',label_status:'reviewed_dashboard',notes:''}; for(const a of ATTRIBUTES){const d=s[`${rowKey(row)}|${a}`]||{}; out[`${a}_truth_choice`]=d.choice||''; out[`${a}_truth_value`]=d.truth_value||''; out[`${a}_evidence_url`]=d.evidence_url||''; out[`${a}_label_source`]=d.reviewer||reviewerName()} return out}).filter(r=>!filterFn||filterFn(r)); return [h.join(','),...rows.map(r=>h.map(k=>csvEsc(r[k]||'')).join(','))].join('\n')}",
            "function download(name,csv){const blob=new Blob([csv],{type:'text/csv'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download=name; a.click(); URL.revokeObjectURL(url)}",
            "function parseCsv(text){const rows=[];let row=[],cell='',q=false; for(let i=0;i<text.length;i++){const ch=text[i]; if(q){if(ch==='"'&&text[i+1]==='"'){cell+='"';i++}else if(ch==='"')q=false;else cell+=ch}else{if(ch==='"')q=true;else if(ch===','){row.push(cell);cell=''}else if(ch==='\n'){row.push(cell);rows.push(row);row=[];cell=''}else if(ch!=='\r')cell+=ch}} row.push(cell); if(row.some(Boolean)||!rows.length)rows.push(row); const h=rows.shift()||[]; return rows.filter(r=>r.some(Boolean)).map(r=>Object.fromEntries(h.map((x,i)=>[x,r[i]||''])))}",
            "function renderAgreement(files){if(!files.length){document.getElementById('agreementSummary').innerHTML='<p class=hint>Upload reviewer CSVs to compute agreement.</p>'; return} const counts={}; for(const a of ATTRIBUTES)counts[a]={agree:0,total:0}; const votes=new Map(); for(const file of files){for(const row of file.rows){const base=`${row.id||''}|${row.base_id||''}`; for(const a of ATTRIBUTES){const v=row[`${a}_truth_value`]||row[`${a}_truth_choice`]; if(!v)continue; const k=`${base}|${a}`; if(!votes.has(k))votes.set(k,new Map()); votes.get(k).set(file.name,v)}}} const bad=[]; for(const [k,m] of votes){const a=k.split('|').pop(); const unique=new Set(m.values()); counts[a].total++; if(unique.size===1)counts[a].agree++; else bad.push([k,[...m.entries()]])} document.getElementById('disagreementCount').textContent=bad.length; document.getElementById('agreementSummary').innerHTML='<table><tr><th>Attribute</th><th>Agreement</th></tr>'+ATTRIBUTES.map(a=>`<tr><td>${a}</td><td>${counts[a].total?Math.round(counts[a].agree/counts[a].total*100):0}% (${counts[a].agree}/${counts[a].total})</td></tr>`).join('')+'</table>'; document.getElementById('disagreementTable').innerHTML=bad.length?'<table><tr><th>Row / Attribute</th><th>Votes</th></tr>'+bad.map(([k,e])=>`<tr><td>${esc(k)}</td><td>${e.map(([n,v])=>`${esc(n)}: ${esc(v)}`).join('<br>')}</td></tr>`).join('')+'</table>':'<p>No reviewer disagreements found.</p>'}",
            "document.getElementById('prevItem').onclick=()=>{saveCurrent(); pointer=Math.max(0,pointer-1); render()}; document.getElementById('nextItem').onclick=()=>{saveCurrent(); pointer=Math.min(queue.length-1,pointer+1); render()}; document.getElementById('showAllToggle').onchange=()=>{pointer=0;render()}; document.getElementById('exportLabels').onclick=()=>download(`project_a_truth_labels_${reviewerName()}.csv`,exportRows()); document.getElementById('exportFollowup').onclick=()=>download(`project_a_followup_${reviewerName()}.csv`,exportRows(r=>Object.keys(r).some(k=>k.endsWith('_truth_choice')&&r[k]==='not_enough_evidence'))); document.getElementById('clearState').onclick=()=>{localStorage.removeItem(stateKey());render()}; document.getElementById('reviewerName').onchange=render; document.getElementById('truthValue').onchange=()=>saveCurrent(); document.getElementById('evidenceUrl').onchange=()=>saveCurrent(); document.getElementById('reviewNotes').onchange=()=>saveCurrent(); document.getElementById('reviewCsvInput').onchange=async e=>{const f=e.target.files?.[0]; if(!f)return; reviewRows=parseCsv(await f.text()); pointer=0; render()}; document.getElementById('multiCsvInput').onchange=async e=>{const files=[...(e.target.files||[])], parsed=[]; for(const f of files)parsed.push({name:f.name,rows:parseCsv(await f.text())}); renderAgreement(parsed)}; document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA'].includes(document.activeElement.tagName))return; if(['1','2','3','4','5'].includes(e.key))choose(CHOICES[Number(e.key)-1]); if(e.key.toLowerCase()==='n'){pointer=Math.min(queue.length-1,pointer+1);render()} if(e.key.toLowerCase()==='b'){pointer=Math.max(0,pointer-1);render()} if(e.key.toLowerCase()==='e')document.getElementById('evidenceUrl').focus()}); render();",
            "</script>",
            "</body></html>",
        ]
    )


def write_truth_validation_dashboard(review_csv: str | Path, output_dir: str | Path, *, limit: int | None = None) -> dict[str, str]:
    rows = load_review_rows(review_csv, limit=limit)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    html_path = output / "truth_validation.html"
    html_path.write_text(
        render_truth_validation_html(TruthDashboardData(review_rows=rows, source_csv=str(review_csv))),
        encoding="utf-8",
    )
    summary_path = output / "truth_validation_summary.json"
    summary_path.write_text(json.dumps(_review_summary(rows), indent=2, sort_keys=True), encoding="utf-8")
    return {"html": str(html_path), "summary": str(summary_path)}
