(() => {
  const $ = (id) => document.getElementById(id);
  const form = $('form'), file = $('file'), analyze = $('analyze'), reset = $('reset');
  const status = $('status'), result = $('result');
  let generation = 0;

  const setStatus = (text, kind) => {
    status.textContent = text;
    status.className = 'status' + (kind ? ' ' + kind : '');
  };

  fetch('/readyz').then(r => r.json()).then(d => {
    $('ready').textContent = d.ready ? `Service ready (${d.detector || 'detector'}).` : 'Service not ready: ' + (d.reasons || []).join('; ');
  }).catch(() => { $('ready').textContent = 'Service unreachable.'; });

  file.addEventListener('change', () => {
    generation++;               // a new file invalidates any pending response
    analyze.disabled = !file.files.length;
    setStatus('');
  });

  reset.addEventListener('click', () => {
    generation++;
    form.reset();
    analyze.disabled = true;
    result.hidden = true;
    setStatus('');
  });

  const drawBoxes = (resp) => {
    const stage = $('stage');
    stage.querySelectorAll('.bbox').forEach(n => n.remove());
    const W = resp.input.width, H = resp.input.height, det = resp.detection;
    (det.candidates || []).forEach((c, i) => {
      const el = document.createElement('div');
      const primary = i === det.selected_index;
      el.className = 'bbox' + (primary ? ' primary' : '');
      const [l, t, r, b] = c.box;
      Object.assign(el.style, { left: (100 * l / W) + '%', top: (100 * t / H) + '%',
        width: (100 * (r - l) / W) + '%', height: (100 * (b - t) / H) + '%' });
      if (primary) {
        const tag = document.createElement('span');
        tag.className = 'tag';
        tag.textContent = `vehicle ${(c.score * 100).toFixed(0)}% (detector)`;
        el.appendChild(tag);
      }
      stage.appendChild(el);
    });
  };

  const fact = (k, v) => `<dt>${k}</dt><dd>${v}</dd>`;

  const render = (resp) => {
    result.hidden = false;
    $('json').textContent = JSON.stringify(resp, null, 2);
    const det = resp.detection;
    const img = $('display');
    img.onload = () => drawBoxes(resp);
    img.src = resp.processing.images['display.jpg'];
    if (resp.status === 'no_vehicle') {
      $('summary').textContent = 'No supported vehicle target found. Try a clearer photo with the whole car visible.';
    } else {
      $('summary').textContent = `Vehicle detected (detector score ${det.detector_score.toFixed(2)}).`;
    }
    const t = resp.timings;
    $('facts').innerHTML = [
      fact('Detector', det.detector_version),
      fact('Vehicle candidates', `${det.candidate_count} (threshold ${det.threshold} for selection)`),
      fact('Selected box (px)', det.selected_box ? det.selected_box.map(v => v.toFixed(0)).join(', ') : 'none'),
      fact('Image', `${resp.input.width} x ${resp.input.height}${resp.input.resized_from ? ' (resized from ' + resp.input.resized_from.join(' x ') + ')' : ''}`),
      fact('Server time', `${t.total_ms} ms (decode ${t.decode_ms}, detection ${t.detection_ms}, queue ${t.queue_ms})`),
    ].join('');
    const crop = resp.processing.images['crop.jpg'];
    $('crop-wrap').hidden = !crop;
    if (crop) $('crop').src = crop;
  };

  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    if (!file.files.length) return;
    const my = ++generation;
    analyze.disabled = true;
    setStatus('Uploading and analyzing: decoding, detecting vehicles...', 'busy');
    const fd = new FormData();
    fd.append('file', file.files[0]);
    const t0 = performance.now();
    try {
      const r = await fetch('/predict', { method: 'POST', body: fd });
      const resp = await r.json();
      if (my !== generation) return;          // stale response for an older selection
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      if (resp.error) {
        setStatus(resp.error.message, 'error');
        result.hidden = true;
      } else {
        setStatus(`Done in ${secs} s.`);
        render(resp);
      }
    } catch (e) {
      if (my !== generation) return;
      setStatus('Network error: the server could not be reached. Check your connection and retry.', 'error');
    } finally {
      if (my === generation) analyze.disabled = !file.files.length;
    }
  });
})();
