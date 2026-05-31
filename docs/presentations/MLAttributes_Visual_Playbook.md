---
marp: true
title: MLAttributes Visual Playbook
paginate: true
theme: default
style: |
  section {
    font-family: "Aptos", "Inter", "Segoe UI", Arial, sans-serif;
    background: linear-gradient(180deg, #fdfaff 0%, #f7f2ff 100%);
    color: #23173b;
  }
  h1, h2, h3 {
    color: #4c1d95;
  }
  h1 {
    letter-spacing: -0.03em;
  }
  .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    font-weight: 800;
    color: #7c3aed;
    margin-bottom: 10px;
  }
  .muted {
    color: #6b5b95;
  }
  .grid2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
  }
  .grid3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }
  .card {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(109,40,217,0.16);
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: 0 10px 24px rgba(76,29,149,0.08);
  }
  .card h3 {
    margin-top: 0;
    margin-bottom: 10px;
  }
  .callout {
    border-left: 4px solid #7c3aed;
    background: rgba(124,58,237,0.08);
    padding: 12px 14px;
    border-radius: 12px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(109,40,217,0.14);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 10px 24px rgba(76,29,149,0.06);
  }
  th, td {
    padding: 10px 12px;
    border-bottom: 1px solid rgba(109,40,217,0.10);
    text-align: left;
    vertical-align: top;
  }
  th {
    background: #f4ebff;
    color: #4c1d95;
    font-weight: 800;
  }
  .flow {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    font-size: 0.88rem;
    line-height: 1.55;
    white-space: pre-wrap;
    margin: 0;
  }
  .ok { color: #166534; font-weight: 700; }
  .warn { color: #a16207; font-weight: 700; }
  .bad { color: #b91c1c; font-weight: 700; }
---

# MLAttributes Visual Playbook
## How to make the important PAC ideas readable

- Anthony Martinez - Overture/CRWN102
- One concept per slide
- One diagram per idea
- One caveat per result

---

# The visual rule

If a slide needs more than one reading path, it is doing too much.

<div class="grid3">
<div class="card">
<h3>1. Show the flow</h3>
<p class="muted">Use arrows when the audience needs to see how data moves.</p>
</div>
<div class="card">
<h3>2. Show the tradeoff</h3>
<p class="muted">Use side-by-side cards when two metrics must be compared.</p>
</div>
<div class="card">
<h3>3. Show the portfolio</h3>
<p class="muted">Use a matrix when multiple corpora or methods each solve different weaknesses.</p>
</div>
</div>

---

# Core system flow

<div class="card">
<pre class="flow">dataset / replay
      |
      v
retrieval + dorking
      |
      v
claim extraction
      |
      v
EvidenceGraph
  |      |      |
  |      |      +-- contradiction / stale / identity checks
  |      +--------- claim grouping by normalized value
  +----------------- source authority / freshness / relevance
      |
      v
resolver v2
  |      \
  |       \-- abstain when the evidence is weak
  v
benchmarks / dashboard / slide deck</pre>
</div>

<div class="callout">
The audience should be able to point at the slide and say: "this is where evidence becomes claims, and this is where claims become a decision."
</div>

---

# Claim-level decision

<div class="grid2">
<div class="card">
<h3>Supported claim</h3>
<p><strong>Official site</strong>: fresh, place-relevant, same entity.</p>
<p><strong>Government registry</strong>: corroborates the same normalized value.</p>
<p><strong>Decision</strong>: keep the claim and explain why.</p>
</div>
<div class="card">
<h3>Rejected claim</h3>
<p><strong>Aggregator</strong>: stale, lower authority, or wrong branch.</p>
<p><strong>Generic homepage</strong>: not actually about the place.</p>
<p><strong>Decision</strong>: reject or abstain if nothing better exists.</p>
</div>
</div>

<div class="card">
<pre class="flow">source pages -> extracted claims -> grouped by normalized value
        |                 |                 |
        |                 |                 +-- same value? cluster it
        |                 +-------------------- contradiction? mark it
        +-------------------------------------- authority / freshness / identity score</pre>
</div>

---

# Replay portfolio matrix

<table>
  <thead>
    <tr>
      <th>Corpus</th>
      <th>Best visual message</th>
      <th>What it proves</th>
      <th>Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>pac_hard_cases_replay</code></td>
      <td>Abstention under noisy evidence</td>
      <td class="ok">Best abstention-heavy mixed-evidence diagnostic</td>
      <td class="warn">Small, curated</td>
    </tr>
    <tr>
      <td><code>santa_cruz_challenge_replay</code></td>
      <td>Dense authority-page proof</td>
      <td class="ok">All-core attributes covered</td>
      <td class="warn">Still Santa Cruz-shaped</td>
    </tr>
<tr>
      <td><code>pac_promoted_replay</code></td>
      <td>Balanced mixed proof</td>
      <td class="ok">Phone, address, and identity drift all appear</td>
      <td class="warn">Not the full generalization ceiling</td>
    </tr>
    <tr>
      <td><code>pac_contact_replay</code></td>
      <td>Contact-heavy phone/address proof</td>
      <td class="ok">Strongest checked-in slice for contact claims and abstention</td>
      <td class="warn">Still narrower than the full collected portfolio</td>
    </tr>
    <tr>
      <td><code>collected_mixed_generalization_replay</code></td>
      <td>Collected generalization</td>
      <td class="ok">Broadest checked-in evidence portfolio</td>
      <td class="warn">Phone/address gaps remain</td>
    </tr>
  </tbody>
</table>

---

# v5 vs v6

<div class="grid2">
<div class="card">
<h3>v5</h3>
<p><strong>94.0%</strong> expected-behavior accuracy</p>
<p><strong>10.26%</strong> unsafe predictions</p>
<p class="muted">Use when you want the stronger raw score on the mixed collected benchmark.</p>
</div>
<div class="card">
<h3>v6</h3>
<p><strong>72.5%</strong> expected-behavior accuracy</p>
<p><strong>0.0%</strong> unsafe predictions</p>
<p class="muted">Use when safety and abstention discipline matter more than raw score.</p>
</div>
</div>

<div class="callout">
The clearest explanation is not "which model won?" It is "which tradeoff does this model make?"
</div>

---

# How Sure improves under PAC

<div class="grid2">
<div class="card">
<h3>Sure alone</h3>
<pre class="flow">cheap heuristic
   |
   v
pick one value</pre>
<p class="bad">Fast, but it can over-trust stale or wrong-entity evidence.</p>
</div>
<div class="card">
<h3>Sure inside MLAttributes</h3>
<pre class="flow">cheap heuristic
   |
   v
authority gate + freshness gate + identity gate
   |
   v
confidence threshold + abstain</pre>
<p class="ok">Still simple, but now safer and more useful on hard cases.</p>
</div>
</div>

<div class="card">
<strong>Presentation line:</strong> the PAC system does not replace the traditional baseline; it gives the baseline better evidence to work with.
</div>

---

# What each visual should answer

<table>
  <thead>
    <tr>
      <th>Question</th>
      <th>Best visual form</th>
      <th>Example from MLAttributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>How does the system work?</td>
      <td>Flow diagram</td>
      <td>dataset -> retrieval -> claim extraction -> EvidenceGraph -> resolver</td>
    </tr>
    <tr>
      <td>Why does it abstain?</td>
      <td>Two-column decision card</td>
      <td>supported claim vs stale or wrong-entity claim</td>
    </tr>
    <tr>
      <td>What corpora matter?</td>
      <td>Portfolio matrix</td>
      <td>hard cases, Santa Cruz, promoted mixed, collected mixed</td>
    </tr>
    <tr>
      <td>Is v5 or v6 better?</td>
      <td>Tradeoff card</td>
      <td>v5 score vs v6 safety</td>
    </tr>
    <tr>
      <td>How does Sure fit?</td>
      <td>Before / after lane</td>
      <td>baseline only vs baseline + PAC gates</td>
    </tr>
  </tbody>
</table>

---

# Slide template

Use this exact structure when the concept is important:

<div class="grid3">
<div class="card">
<h3>1. Headline</h3>
<p class="muted">Say the point in one sentence.</p>
</div>
<div class="card">
<h3>2. Diagram</h3>
<p class="muted">Show the flow, comparison, or portfolio.</p>
</div>
<div class="card">
<h3>3. Caveat</h3>
<p class="muted">One line on what the visual does not prove.</p>
</div>
</div>

<div class="callout">
If the audience can explain the slide back to you after one pass, the visual is doing its job.
</div>
