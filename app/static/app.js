(() => {
  const $ = (id) => document.getElementById(id);
  const form = $('form'), file = $('file'), analyze = $('analyze'), reset = $('reset'), status = $('status');
  const cards = ['result', 'verdict', 'pred-card', 'passport-card', 'challenge-card', 'ref-card', 'proc-card', 'cmp-card', 'record-card'];
  let generation = 0, current = null, challenge = null, showAll = false, challengeBusy = false;

  const setStatus = (el, text, kind) => { el.textContent = text; el.className = 'status' + (kind ? ' ' + kind : ''); };
  const el = (tag, text, cls) => { const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e; };
  const hideAll = () => cards.forEach(c => { $(c).hidden = true; });
  const pct = (v) => (100 * v).toFixed(1) + '%';

  fetch('/readyz').then(r => r.json()).then(d => {
    if (!d.ready) { $('ready').textContent = 'Service not ready: ' + (d.reasons || []).join('; '); $('ready').className = 'ready off'; return; }
    $('ready').textContent = d.recognition ? 'Online' : 'Online · detection only';
    $('ready').className = 'ready on';
    $('ready').title = `${d.backbone || ''} · ${d.calibration_kind || ''} calibration · ${d.detector} · bundle ${d.bundle_id}`;
    if (!d.recognition) $('modes').hidden = true;
  }).catch(() => { $('ready').textContent = 'Service unreachable.'; $('ready').className = 'ready off'; });

  const drop = $('drop'), dropName = $('drop-name'), dropHint = dropName.textContent;
  const dropText = $('drop-text'), dropTextHtml = dropText.innerHTML, thumb = $('drop-thumb');
  let thumbUrl = null;
  const showFile = () => {
    const f = file.files[0];
    drop.classList.toggle('has-file', !!f);
    if (thumbUrl) { URL.revokeObjectURL(thumbUrl); thumbUrl = null; }
    if (f && f.type.startsWith('image/')) {
      thumbUrl = URL.createObjectURL(f);
      thumb.src = thumbUrl; thumb.hidden = false;
    } else { thumb.hidden = true; thumb.removeAttribute('src'); }
    if (f) { dropText.textContent = f.name; dropName.textContent = `${(f.size / 1048576).toFixed(1)} MB · ready to analyze · click to change`; }
    else { dropText.innerHTML = dropTextHtml; dropName.textContent = dropHint; }
  };
  thumb.addEventListener('error', () => { thumb.hidden = true; });
  file.addEventListener('change', () => { generation++; analyze.disabled = !file.files.length; setStatus(status, ''); showFile(); });
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, () => drop.classList.remove('over')));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f) return;
    const dt = new DataTransfer(); dt.items.add(f); file.files = dt.files;
    file.dispatchEvent(new Event('change', { bubbles: true }));
  });
  // The button's id "reset" shadows form.reset (named form controls become form properties), so call the native method.
  reset.addEventListener('click', () => {
    generation++; HTMLFormElement.prototype.reset.call(form); analyze.disabled = true; hideAll();
    setStatus(status, ''); current = null; challenge = null; showFile();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
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
    if (cp && cp.box) add(cp.box, true, cp.fallback ? 'no contour box · whole image analysed' : `classical proposal ${cp.score.toFixed(2)}`);
  };

  const labelFor = (resp, detScore) => {
    const p = resp.prediction;
    const base = `vehicle ${(detScore * 100).toFixed(0)}% det`;
    if (!p.available || !p.top5.length) return base;
    const top = p.top5[0];
    const st = resp.passport && resp.passport.decision_state;
    if (st === 'rejected' || st === 'inadequate') return `not verified · ${base}`;
    return `${top.name} · ${pct(top.score)}`;
  };

  const renderPrediction = (resp) => {
    const p = resp.prediction, pp = resp.passport;
    $('pred-card').hidden = !p.available;
    if (!p.available) return;
    const rejected = pp.decision_state === 'rejected';
    $('pred-h').textContent = rejected ? 'Diagnostic guesses, unverified' : 'Best guess (ordinary prediction)';
    $('pred-label').textContent = p.calibrated ? 'Calibrated confidence' : 'Uncalibrated score (comparison mode)';
    const ol = $('top5'); ol.replaceChildren();
    p.top5.forEach(c => {
      const li = el('li');
      li.style.setProperty('--p', Math.max(0, Math.min(1, c.score)));
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
      ? 'Development calibration (tuning data). The 90% target is a population-level setting, not the probability that this result is correct, and no coverage claim applies yet.'
      : (ms.calibration_kind === 'final'
        ? 'Calibrated on 815 held-out images to contain the true car 90% of the time (for the 196 supported models).' : '');
    $('pp-dev').hidden = !$('pp-dev').textContent;
    let text = STATE_TEXT[pp.decision_state] || pp.decision_state;
    const spec = pp.specificity;
    if (pp.decision_state === 'exact_candidate' && pp.accepted_identity) text += `: ${pp.accepted_identity.name}. This can still be wrong.`;
    if (pp.decision_state === 'ambiguous' && spec) text += spec.level === 'family' ? `. Supported answer: ${spec.label} family.` : `. Supported answer: make ${spec.label}.`;
    $('pp-state').textContent = text;
    const reasons = [];
    if (spec && spec.unresolved && spec.unresolved.length) reasons.push(spec.unresolved.join('; '));
    const REASON = { max_probability_below_reject_threshold: 'the top score is below the reject threshold',
      empty_candidate_set: 'no category passed the calibrated threshold', no_shared_make: 'the remaining candidates differ in make',
      no_shared_family: 'the remaining candidates differ in model', no_vehicle: 'no vehicle was detected',
      no_eligible_detection: 'no car, bus or truck was detected', no_prediction: 'no prediction was made',
      no_matching_calibration_bundle: 'this mode has no calibration', specificity_make: 'only the make is certain',
      specificity_family: 'only the make and model are certain' };
    if (pp.reason_codes && pp.reason_codes.length) reasons.push('Why: ' + pp.reason_codes.map(c => REASON[c] || c.replace(/_/g, ' ')).join('; '));
    $('pp-reason').textContent = reasons.join('. ');
    const cands = pp.candidates || [];
    const failure = pp.candidate_set_is_failure_convention;
    $('cand-wrap').hidden = pp.decision_state === 'comparison_only';
    $('cand-h').textContent = failure ? 'Candidate set: all 196 categories (failure convention)' : `Candidate set: ${pp.candidate_count} categor${pp.candidate_count === 1 ? 'y' : 'ies'}`;
    const ol = $('cands'); ol.replaceChildren();
    (showAll ? cands : cands.slice(0, 5)).forEach(c => {
      const li = el('li');
      li.style.setProperty('--p', Math.max(0, Math.min(1, c.score)));
      li.appendChild(el('span', `${c.make} ${c.model}, ${c.year}`));
      li.appendChild(el('span', pct(c.score), 'score'));
      ol.appendChild(li);
    });
    $('show-all').hidden = cands.length <= 5;
    $('show-all').textContent = showAll ? 'Show first 5' : `Show all ${cands.length}`;
  };
  $('show-all').addEventListener('click', () => { showAll = !showAll; if (current) renderPassport(current); });

  const VERDICT = {
    exact_candidate: ['ok', 'Identified'],
    ambiguous: ['partial', 'Narrowed down'],
    inadequate: ['partial', 'Not enough evidence'],
    rejected: ['bad', 'Not verified'],
    detection_failed: ['bad', 'No vehicle found'],
    comparison_only: ['info', 'Comparison mode'],
    unavailable: ['info', 'Detection only'],
  };
  const renderVerdict = (resp) => {
    const v = $('verdict'), pp = resp.passport || {}, p = resp.prediction || {};
    const state = resp.status === 'no_vehicle' ? 'detection_failed' : pp.decision_state;
    const [kind, chip] = VERDICT[state] || ['info', state || 'Result'];
    const top = p.available && p.top5 && p.top5.length ? p.top5[0] : null;
    let title = '', sub = '', meter = top ? top.score : 0;
    const spec = pp.specificity;
    if (state === 'exact_candidate' && pp.accepted_identity) {
      title = pp.accepted_identity.name;
      sub = `Calibrated score ${pct(top ? top.score : 0)}. The candidate set holds this category only; it can still be wrong.`;
    } else if (state === 'ambiguous' && spec) {
      title = spec.level === 'family' ? `${spec.label}` : `Make: ${spec.label}`;
      sub = `${pp.candidate_count} categories remain` + (spec.unresolved && spec.unresolved.length ? ` (${spec.unresolved.join('; ')})` : '') + '. The answer stops where the evidence does.';
    } else if (state === 'inadequate') {
      title = 'Several makes remain';
      sub = top ? `Best guess ${top.name} (${pct(top.score)}) is not supported by the candidate set.` : 'The candidate set spans more than one make.';
    } else if (state === 'rejected') {
      title = 'Could not verify a supported category';
      sub = top ? `Best guess ${top.name} scored ${pct(top.score)}, below the reject threshold. It may be a model outside the 196 categories.` : '';
    } else if (state === 'detection_failed') {
      title = 'No car, bus or truck detected';
      sub = 'Try a clearer photo with the whole vehicle visible.'; meter = 0;
    } else if (state === 'comparison_only' && top) {
      title = top.name;
      sub = `Comparison mode (${resp.active_mode}): uncalibrated score ${pct(top.score)}. Switch to Hybrid for the Passport.`;
    } else if (top) {
      title = top.name; sub = `Score ${pct(top.score)}.`;
    } else {
      title = 'Vehicle detected'; sub = p.unavailable_reason || ''; meter = 0;
    }
    v.dataset.kind = kind;
    $('v-chip').textContent = chip;
    $('v-title').textContent = title;
    $('v-sub').textContent = sub;
    $('v-meter').style.width = `${Math.round(100 * Math.max(0, Math.min(1, meter)))}%`;
    v.hidden = false;
  };

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

  const renderRefs = (resp) => {
    const r = resp.references || {};
    $('ref-card').hidden = !(r.state === 'available' && r.items && r.items.length);
    if ($('ref-card').hidden) return;
    const ol = $('refs'); ol.replaceChildren();
    r.items.forEach((it, i) => {
      const li = el('li');
      li.appendChild(el('span', `${it.name} (training photo ${it.sample_key})`));
      li.appendChild(el('span', `similarity ${it.cosine_similarity.toFixed(3)}`, 'score'));
      if (r.agrees_with_best_guess && !r.agrees_with_best_guess[i]) li.appendChild(el('div', 'Differs from the best guess.', 'muted small'));
      ol.appendChild(li);
    });
    $('ref-note').textContent = 'Training photos are not shown (Stanford Cars is licensed for non-commercial research only).';
  };

  const render = (resp) => {
    current = resp; challenge = null; showAll = false;
    $('result').hidden = false;
    $('record-card').hidden = false;
    if (resp.prediction && resp.prediction.available) { $('cmp-card').hidden = false; resetCompare(); loadTestTable(); }
    $('json').textContent = JSON.stringify(resp, null, 2);
    const det = resp.detection;
    const img = $('display');
    img.onload = () => drawBoxes(resp);
    img.src = resp.processing.images['display.jpg'];
    const p = resp.prediction;
    if (resp.status === 'no_vehicle') {
      $('summary').textContent = 'No car, bus or truck found. Try a clearer photo with the whole vehicle visible.';
    } else if (p.available && p.top5.length) {
      const t = p.top5[0];
      const unver = ['rejected', 'inadequate'].includes(resp.passport && resp.passport.decision_state);
      $('summary').textContent = (unver ? 'Vehicle found; identity not verified' : 'Vehicle found') +
        (det.detector_score != null ? ` · detector confidence ${(100 * det.detector_score).toFixed(0)}%` : '');
    } else {
      $('summary').textContent = `Vehicle detected (detector score ${det.detector_score.toFixed(2)}). ${p.unavailable_reason || ''}`;
    }
    $('caption').textContent = det.classical_proposal
      ? (det.classical_proposal.fallback
          ? 'Classical mode: no contour box passed the size and shape filters, so the whole image was analysed. Thin boxes are YOLO detections.'
          : 'Classical mode: thick box from the contour proposal; thin boxes are YOLO detections.')
      : 'Thick box: the vehicle analysed (largest one found). Thin boxes: other vehicles.';
    const t = resp.timings;
    const facts = [
      ['Mode', resp.active_mode + (resp.active_mode === resp.production_mode ? ' (Passport production mode)' : ' (comparison mode)')],
      ['Detector', `${det.detector_version}, ${det.candidate_count} vehicle candidate(s), threshold ${det.threshold}`],
      ['Image', `${resp.input.width} x ${resp.input.height}` + (resp.input.resized_from ? ` (resized from ${resp.input.resized_from.join(' x ')})` : '')],
      ['Server time', `${t.total_ms} ms` + Object.entries(t).filter(([k]) => k !== 'total_ms').map(([k, v]) => `, ${k.replace('_ms', '')} ${v}`).join('')],
    ];
    $('facts').replaceChildren(...facts.flatMap(([k, v]) => [el('dt', k), el('dd', v)]));
    renderVerdict(resp); renderPrediction(resp); renderPassport(resp); renderRefs(resp); renderProcessing(resp);
    if (window.innerWidth < 960) $('verdict').scrollIntoView({ behavior: 'smooth', block: 'start' });
    const chAvail = resp.challenge && resp.challenge.state === 'not_requested';
    $('challenge-card').hidden = !chAvail;
    $('ch-result').hidden = true; setStatus($('ch-status'), ''); $('run-challenge').disabled = false;
  };

  document.querySelectorAll('button.ex').forEach(b => b.addEventListener('click', async () => {
    try {
      const r = await fetch(`/static/examples/${b.dataset.ex}.jpg`);
      if (!r.ok) throw new Error('example not found');
      const f = new File([await r.blob()], `${b.dataset.ex}.jpg`, { type: 'image/jpeg' });
      const dt = new DataTransfer(); dt.items.add(f); file.files = dt.files;
      file.dispatchEvent(new Event('change', { bubbles: true }));
      form.requestSubmit();
    } catch (e) { setStatus(status, 'Could not load the example photo. Choose your own photo instead.', 'error'); }
  }));

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

  // ---------------- Model comparison (same photo through all three modes + saved test results)
  const MODE_NAMES = { classical: 'Classical', deep: 'Deep', hybrid: 'Hybrid' };
  let cmpTestLoaded = false;
  const loadTestTable = async () => {
    if (cmpTestLoaded) return;
    try {
      const b = await (await fetch('/benchmarks')).json();
      const rep = b.reports[`e2e_test_${b.bundle_id}.json`];
      if (!rep) { $('cmp-test').replaceChildren(el('tr')); $('cmp-test-cap').textContent = 'Official test results are not available for this build.'; return; }
      $('cmp-test-cap').textContent = `Official test set (${rep.population.count.toLocaleString('en')} unseen images)`;
      $('cmp-test').replaceChildren(...['classical', 'deep', 'hybrid'].filter(m => rep.modes[m]).map(m => {
        const r = rep.modes[m], tr = el('tr');
        [MODE_NAMES[m], pct(r.end_to_end_top1.value), pct(r.end_to_end_top5.value),
         r.localization.mean_iou_valid_selected.value.toFixed(3), r.fine_grained_ap.AP50.toFixed(3)].forEach(v => tr.appendChild(el('td', v)));
        return tr;
      }));
      cmpTestLoaded = true;
    } catch (e) { $('cmp-test-cap').textContent = 'Could not load the saved test results.'; }
  };
  const resetCompare = () => {
    $('cmp-rows').replaceChildren(Object.assign(el('tr'), { innerHTML: '<td colspan="5" class="muted">Not run yet.</td>' }));
    setStatus($('cmp-status'), '');
  };
  $('run-compare').addEventListener('click', async () => {
    if (!file.files.length) return;
    const my = generation, btn = $('run-compare');
    btn.disabled = true;
    const rows = [];
    try {
      for (const m of ['classical', 'deep', 'hybrid']) {
        setStatus($('cmp-status'), `Running ${MODE_NAMES[m]}...`, 'busy');
        const fd = new FormData(); fd.append('file', file.files[0]); fd.append('mode', m);
        const t0 = performance.now();
        const r = await (await fetch('/predict', { method: 'POST', body: fd })).json();
        if (my !== generation) return;
        const tr = el('tr', null, current && current.active_mode === m ? 'active' : '');
        const p = r.prediction || {}, cp = r.detection && r.detection.classical_proposal;
        const top = p.available && p.top5 && p.top5.length ? p.top5[0] : null;
        const st = r.passport && r.passport.decision_state;
        const answer = r.error ? r.error.message : !top ? 'No vehicle found'
          : top.name + (['rejected', 'inadequate'].includes(st) ? ' (not verified)' : '');
        const region = m === 'classical' ? (cp && cp.fallback ? 'whole image (no contour box)' : 'contour proposal')
          : (r.detection && r.detection.selected_box ? 'YOLO11n box' : 'none');
        [MODE_NAMES[m], answer, top ? pct(top.score) + (p.calibrated ? ' cal.' : '') : '–', region,
         `${((performance.now() - t0) / 1000).toFixed(1)} s`].forEach(v => tr.appendChild(el('td', v)));
        rows.push(tr);
        $('cmp-rows').replaceChildren(...rows);
      }
      setStatus($('cmp-status'), 'Done. The highlighted row is the model shown above.');
    } catch (e) {
      if (my === generation) setStatus($('cmp-status'), 'Network error while comparing. Retry.', 'error');
    } finally { btn.disabled = false; }
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
