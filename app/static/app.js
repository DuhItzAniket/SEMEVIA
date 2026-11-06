(() => {
  const $ = (id) => document.getElementById(id);
  const form = $('form'), file = $('file'), analyze = $('analyze'), reset = $('reset'), status = $('status');
  const cards = ['result', 'pred-card', 'passport-card', 'challenge-card', 'proc-card', 'record-card'];
  let generation = 0, current = null, challenge = null, showAll = false, challengeBusy = false;

  const setStatus = (el, text, kind) => { el.textContent = text; el.className = 'status' + (kind ? ' ' + kind : ''); };
  const el = (tag, text, cls) => { const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e; };
  const hideAll = () => cards.forEach(c => { $(c).hidden = true; });
  const pct = (v) => (100 * v).toFixed(1) + '%';

  fetch('/readyz').then(r => r.json()).then(d => {
    if (!d.ready) { $('ready').textContent = 'Service not ready: ' + (d.reasons || []).join('; '); return; }
    $('ready').textContent = `Service ready. Detector ${d.detector}. ` + (d.recognition ? `Modes: ${d.modes.join(', ')}.` : '');
    const note = $('build-note');
    note.hidden = false;
    note.textContent = d.recognition
      ? `Model: ${d.backbone}; ${d.calibration_kind} calibration. Predictions are real model outputs, but this is not the final fine-tuned release.`
      : `Development build: vehicle detection only. ${d.recognition_reason || ''}`;
    if (!d.recognition) $('modes').hidden = true;
  }).catch(() => { $('ready').textContent = 'Service unreachable.'; });

  file.addEventListener('change', () => { generation++; analyze.disabled = !file.files.length; setStatus(status, ''); });
  reset.addEventListener('click', () => { generation++; form.reset(); analyze.disabled = true; hideAll(); setStatus(status, ''); current = null; challenge = null; });
  document.querySelectorAll('input[name=mode]').forEach(r => r.addEventListener('change', () => { generation++; }));

  const drawBoxes = (resp) => {
    const stage = $('stage');
    stage.querySelectorAll('.bbox').forEach(n => n.remove());
    const W = resp.input.width, H = resp.input.height, det = resp.detection;
    const add = (box, primary, label) => {
      const d = el('div', null, 'bbox' + (primary ? ' primary' : ''));
      const [l, t, r, b] = box;
      Object.assign(d.style, { left: (100 * l / W) + '%', top: (100 * t / H) + '%', width: (100 * (r - l) / W) + '%', height: (100 * (b - t) / H) + '%' });
      if (label) d.appendChild(el('span', label, 'tag'));
      stage.appendChild(d);
    };
    const cp = det.classical_proposal;
    (det.candidates || []).forEach((c, i) => add(c.box, !cp && i === det.selected_index,
      !cp && i === det.selected_index ? labelFor(resp, c.score) : null));
    if (cp && cp.box) add(cp.box, true, `classical proposal ${cp.score.toFixed(2)}`);
  };

  const labelFor = (resp, detScore) => {
    const p = resp.prediction;
    const base = `vehicle ${(detScore * 100).toFixed(0)}% det`;
    if (!p.available || !p.top5.length) return base;
    const top = p.top5[0];
    return `${top.name} (${pct(top.score)} cls) | ${base}`;
  };

  const renderPrediction = (resp) => {
    const p = resp.prediction, pp = resp.passport;
    $('pred-card').hidden = !p.available;
    if (!p.available) return;
    const rejected = pp.decision_state === 'rejected';
    $('pred-h').textContent = rejected ? '2. Diagnostic guesses, unverified' : '2. Best guess (ordinary prediction)';
    $('pred-label').textContent = `${p.score_type}. Classification score is separate from the detector score.`;
    const ol = $('top5'); ol.replaceChildren();
    p.top5.forEach(c => {
      const li = el('li');
      li.appendChild(el('span', `${c.make} ${c.model}, ${c.year}`));
      li.appendChild(el('span', pct(c.score), 'score'));
      ol.appendChild(li);
    });
  };

  const STATE_TEXT = {
    exact_candidate: 'One category in the candidate set',
    ambiguous: 'Several categories remain',
    rejected: 'Could not verify a supported category',
    inadequate: 'Insufficient evidence for a narrower answer',
    detection_failed: 'No supported vehicle target found',
    comparison_only: 'Comparison mode: Passport calibration unavailable',
    unavailable: 'Passport unavailable in this build',
  };

  const renderPassport = (resp) => {
    const pp = resp.passport;
    const card = $('passport-card');
    card.hidden = !resp.prediction.available && pp.decision_state !== 'detection_failed';
    if (card.hidden) return;
    const ms = resp.model_status || {};
    $('pp-dev').textContent = ms.calibration_kind === 'development'
      ? 'Development calibration (tuning data). The 90% target is a population-level setting, not the probability that this result is correct, and no coverage claim applies yet.' : '';
    $('pp-dev').hidden = !$('pp-dev').textContent;
    let text = STATE_TEXT[pp.decision_state] || pp.decision_state;
    const spec = pp.specificity;
    if (pp.decision_state === 'exact_candidate' && pp.accepted_identity) text += `: ${pp.accepted_identity.name}. This can still be wrong.`;
    if (pp.decision_state === 'ambiguous' && spec) text += spec.level === 'family' ? `. Supported answer: ${spec.label} family.` : `. Supported answer: make ${spec.label}.`;
    $('pp-state').textContent = text;
    const reasons = [];
    if (spec && spec.unresolved && spec.unresolved.length) reasons.push(spec.unresolved.join('; '));
    if (pp.reason_codes && pp.reason_codes.length) reasons.push('Reasons: ' + pp.reason_codes.join(', '));
    $('pp-reason').textContent = reasons.join('. ');
    const cands = pp.candidates || [];
    const failure = pp.candidate_set_is_failure_convention;
    $('cand-wrap').hidden = pp.decision_state === 'comparison_only';
    $('cand-h').textContent = failure ? 'Candidate set: all 196 categories (failure convention)' : `Candidate set: ${pp.candidate_count} categor${pp.candidate_count === 1 ? 'y' : 'ies'}`;
    const ol = $('cands'); ol.replaceChildren();
    (showAll ? cands : cands.slice(0, 5)).forEach(c => {
      const li = el('li');
      li.appendChild(el('span', `${c.make} ${c.model}, ${c.year}`));
      li.appendChild(el('span', pct(c.score), 'score'));
      ol.appendChild(li);
    });
    $('show-all').hidden = cands.length <= 5;
    $('show-all').textContent = showAll ? 'Show first 5' : `Show all ${cands.length}`;
  };
  $('show-all').addEventListener('click', () => { showAll = !showAll; if (current) renderPassport(current); });

  const renderProcessing = (resp) => {
    const imgs = resp.processing.images || {};
    $('proc-card').hidden = !imgs['crop.jpg'];
    if (!imgs['crop.jpg']) return;
    $('v-crop').src = imgs['crop.jpg'];
    [['v-gray', 'gray.png'], ['v-edges', 'edges.png'], ['v-contours', 'contours.png']].forEach(([id, k]) => {
      $(id).closest('figure').hidden = !imgs[k]; if (imgs[k]) $(id).src = imgs[k];
    });
    const m = resp.processing.measurements || {};
    $('measurements').replaceChildren(...Object.entries(m).flatMap(([k, v]) => [el('dt', k.replace(/_/g, ' ')), el('dd', String(v))]));
  };

  const render = (resp) => {
    current = resp; challenge = null; showAll = false;
    $('result').hidden = false;
    $('record-card').hidden = false;
    $('json').textContent = JSON.stringify(resp, null, 2);
    const det = resp.detection;
    const img = $('display');
    img.onload = () => drawBoxes(resp);
    img.src = resp.processing.images['display.jpg'];
    const p = resp.prediction;
    if (resp.status === 'no_vehicle') {
      $('summary').textContent = 'No supported vehicle target found. Try a clearer photo with the whole car visible.';
    } else if (p.available && p.top5.length) {
      const t = p.top5[0];
      $('summary').textContent = `${t.make} ${t.model}, ${t.year}: classification score ${pct(t.score)}` +
        (det.detector_score != null ? `; detector score ${det.detector_score.toFixed(2)}` : '') + '.';
    } else {
      $('summary').textContent = `Vehicle detected (detector score ${det.detector_score.toFixed(2)}). ${p.unavailable_reason || ''}`;
    }
    $('caption').textContent = det.classical_proposal
      ? 'Classical mode: the analyzed region (thick box) comes from the contour-based proposal, not the neural detector. YOLO detections are drawn thin for comparison.'
      : 'One primary vehicle is analyzed: the largest detected car, bus or truck above the detector threshold (thick box). Other candidates are drawn thin.';
    const t = resp.timings;
    const facts = [
      ['Mode', resp.active_mode + (resp.active_mode === resp.production_mode ? ' (Passport production mode)' : ' (comparison mode)')],
      ['Detector', `${det.detector_version}, ${det.candidate_count} vehicle candidate(s), threshold ${det.threshold}`],
      ['Image', `${resp.input.width} x ${resp.input.height}` + (resp.input.resized_from ? ` (resized from ${resp.input.resized_from.join(' x ')})` : '')],
      ['Server time', `${t.total_ms} ms` + Object.entries(t).filter(([k]) => k !== 'total_ms').map(([k, v]) => `, ${k.replace('_ms', '')} ${v}`).join('')],
    ];
    $('facts').replaceChildren(...facts.flatMap(([k, v]) => [el('dt', k), el('dd', v)]));
    renderPrediction(resp); renderPassport(resp); renderProcessing(resp);
    const chAvail = resp.challenge && resp.challenge.state === 'not_requested';
    $('challenge-card').hidden = !chAvail;
    $('ch-result').hidden = true; setStatus($('ch-status'), ''); $('run-challenge').disabled = false;
  };

  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    if (!file.files.length) return;
    const my = ++generation;
    analyze.disabled = true;
    setStatus(status, 'Uploading and analyzing: decoding, detecting the vehicle, extracting features, classifying...', 'busy');
    const fd = new FormData();
    fd.append('file', file.files[0]);
    fd.append('mode', (document.querySelector('input[name=mode]:checked') || {}).value || '');
    const t0 = performance.now();
    try {
      const r = await fetch('/predict', { method: 'POST', body: fd });
      const resp = await r.json();
      if (my !== generation) return;
      if (resp.error) { setStatus(status, resp.error.message, 'error'); hideAll(); return; }
      setStatus(status, `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s.` + (resp.status === 'uncertain' ? ' Result is uncertain; see the Passport.' : ''));
      render(resp);
    } catch (e) {
      if (my !== generation) return;
      setStatus(status, 'Network error: the server could not be reached. Check your connection and retry.', 'error');
    } finally {
      if (my === generation) analyze.disabled = !file.files.length;
    }
  });

  // ---------------- Evidence Challenge
  const selectCell = (idx) => {
    const c = challenge;
    document.querySelectorAll('#ch-buttons button').forEach((b, i) => b.setAttribute('aria-pressed', String(i === idx)));
    document.querySelectorAll('#ch-overlay .ch-cell').forEach((d, i) => d.classList.toggle('sel', i === idx));
    $('ch-img').src = idx == null ? current.processing.images['crop.jpg'] : c.cells[idx].image_url;
    $('ch-img').alt = idx == null ? 'Original crop' : `Crop with the ${c.cells[idx].label} region blurred`;
  };

  const renderChallenge = (c) => {
    challenge = c;
    $('ch-result').hidden = false;
    $('ch-pair').textContent = `Fixed pair: A = ${c.class_a_entry}, B = ${c.class_b_entry}. Original margin m0 = ${c.m0.toFixed(3)} logits` + (c.small_margin ? ' (very small; treat changes with caution).' : '.');
    $('ch-summary').textContent = c.summary;
    const [cw, ch] = [current.processing.crop_bounds[2] - current.processing.crop_bounds[0], current.processing.crop_bounds[3] - current.processing.crop_bounds[1]];
    const ov = $('ch-overlay'); ov.replaceChildren();
    const btns = $('ch-buttons'); btns.replaceChildren();
    const rows = $('ch-rows'); rows.replaceChildren();
    c.cells.forEach((cell, i) => {
      const [l, t, r, b] = cell.bounds;
      const d = el('div', null, 'ch-cell');
      Object.assign(d.style, { left: (100 * l / cw) + '%', top: (100 * t / ch) + '%', width: (100 * (r - l) / cw) + '%', height: (100 * (b - t) / ch) + '%' });
      ov.appendChild(d);
      const sign = cell.delta > 0 ? '+' : '';
      const btn = el('button');
      btn.type = 'button';
      btn.setAttribute('aria-label', `${cell.label}: margin change ${sign}${cell.delta.toFixed(3)}`);
      btn.appendChild(el('span', cell.label)); btn.appendChild(el('br'));
      btn.appendChild(el('span', `${sign}${cell.delta.toFixed(2)}`, cell.delta > 0 ? 'pos' : (cell.delta < 0 ? 'neg' : '')));
      btn.addEventListener('click', () => selectCell(i));
      btns.appendChild(btn);
      const tr = el('tr');
      [cell.label, cell.m_r.toFixed(3), `${sign}${cell.delta.toFixed(3)}`, `${cell.new_top1_entry.name} (${pct(cell.new_top1_prob)})`].forEach(v => tr.appendChild(el('td', v)));
      rows.appendChild(tr);
    });
    $('ch-label').textContent = `${c.label}. Computed in ${c.duration_ms} ms with the active ${c.mode} model.`;
    selectCell(c.selected_cell);
  };
  $('ch-restore').addEventListener('click', () => selectCell(null));

  $('run-challenge').addEventListener('click', async () => {
    if (!current || challengeBusy) return;
    const my = generation, rid = current.request_id;
    challengeBusy = true; $('run-challenge').disabled = true;
    setStatus($('ch-status'), 'Running six blurred-region variants through the same model...', 'busy');
    try {
      const r = await fetch('/challenge', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ request_id: rid }) });
      const c = await r.json();
      if (my !== generation || !current || current.request_id !== rid) return;
      if (c.state !== 'complete') { setStatus($('ch-status'), (c.message || `Challenge ${c.state}: ${c.reason}`) + ' The original result above is unchanged.', 'error'); return; }
      setStatus($('ch-status'), '');
      current.challenge = Object.assign({}, c);
      renderChallenge(c);
    } catch (e) {
      if (my === generation) setStatus($('ch-status'), 'Network error during the challenge. The original result above is unchanged.', 'error');
    } finally {
      challengeBusy = false; if (my === generation) $('run-challenge').disabled = false;
    }
  });

  // ---------------- Passport export (client-side snapshot)
  $('download').addEventListener('click', () => {
    if (!current) return;
    const snap = JSON.parse(JSON.stringify(current));
    snap.updated_at = new Date().toISOString();
    snap.export_note = 'SEMEVIA Passport: reproducibility record, not a signed certificate. Contains hashes and coordinates, not image pixels.';
    if (snap.processing) delete snap.processing.images;
    const blob = new Blob([JSON.stringify(snap, null, 2)], { type: 'application/json' });
    const a = el('a'); a.href = URL.createObjectURL(blob); a.download = `semevia_passport_${snap.request_id}.json`;
    document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
})();
