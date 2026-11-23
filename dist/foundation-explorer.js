(() => {
  const { previews, benchmarks, code } = JSON.parse(
    document.getElementById('foundation-data').textContent
  );
  const comparisons = {
    jobchange: {
      name: 'Job-change intention',
      verdict: 'Random forest led on this holdout',
      takeaway: 'Random forest ranked reported job-change intention better and had lower probability error. Keep it as the benchmark for this task.',
      limit: 'The label records stated intention. The sample does not track actual departures.',
    },
    absence: {
      name: 'Absence episodes',
      verdict: 'TabICLv2 led on all three metrics',
      takeaway: 'The checkpoint improved both ranking and probability accuracy on this split. That makes it worth testing against the same baselines on future periods and employees held out as groups.',
      limit: 'Episodes can repeat for the same employee across partitions. Some features, including the reason for absence, may only be known after an episode starts.',
    },
    productivity: {
      name: 'Team productivity',
      verdict: 'Mixed results for TabICLv2',
      takeaway: 'Average precision was nearly tied (0.927 vs. 0.926). Random forest had higher ROC-AUC and lower probability error, so it remains the stronger starting point here.',
      limit: 'Rows describe teams at different dates. A chronological or team-held-out test would better address performance on future shifts or new teams.',
    },
  };
  const modelNames = {
    logistic: 'Logistic regression', random_forest: 'Random forest',
    extra_trees: 'Extra Trees', hist_gradient: 'Histogram gradient boosting',
    mlp: 'MLP', xgboost: 'XGBoost', lightgbm: 'LightGBM', catboost: 'CatBoost',
    TabICLv2: 'TabICLv2 · foundation model',
  };
  const panel = document.getElementById('foundation-panel');
  const buttons = [...document.querySelectorAll('[data-foundation-dataset]')];
  const escapeHTML = value => String(value ?? '—').replace(/[&<>]/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;',
  })[character]);

  function render(key) {
    const source = previews[key];
    const details = comparisons[key];
    const { models, train_rows, test_rows } = benchmarks[key];
    const metrics = ['roc_auc', 'average_precision', 'brier'];
    const command = 'python dist/run_tabicl.py work/datasets --dataset ' + key;
    panel.innerHTML = `
      <div class="dataset-lab">
        <div class="dataset-lab-head">
          <div><h4>${details.verdict}</h4><p>${train_rows.toLocaleString()} training rows · ${test_rows} held-out rows</p></div>
        </div>
        <div class="table-wrap" tabindex="0" aria-label="Model scores">
          <table class="score-table">
            <thead><tr><th scope="col">Model</th><th scope="col">ROC-AUC ↑</th><th scope="col">Average precision ↑</th><th scope="col">Brier ↓</th></tr></thead>
            <tbody>${models.map(row => `
              <tr class="${row.model === 'TabICLv2' ? 'best' : ''}">
                <th scope="row">${modelNames[row.model]}</th>
                ${metrics.map(metric => `<td>${row.metrics[metric].toFixed(3)}</td>`).join('')}
              </tr>`).join('')}</tbody>
          </table>
        </div>
        <p>${details.takeaway}</p>
        <p class="score-caveat">${details.limit} Results are from one stratified holdout per dataset.</p>
        <details class="code-drawer">
          <summary>Run TabICLv2 on ${details.name.toLowerCase()}</summary>
          <div class="dataset-code-actions"><a href="run_tabicl.py" download>Download Python script</a>
          <button type="button" data-copy-run>Copy Python</button><span role="status"></span></div>
          <pre class="dataset-code"><code>${escapeHTML(code)}</code></pre>
          <pre class="dataset-code"><code>python -m pip install tabicl==2.2.0 pandas scikit-learn
${command}</code></pre>
        </details>
        <details class="code-drawer">
          <summary>Inspect five source rows and provenance</summary>
          <div class="real-table-wrap" tabindex="0" aria-label="Source data preview">
            <p>${source.rows.toLocaleString()} source rows · ${source.columns} columns. ${escapeHTML(source.definition)}</p>
            <table class="real-table"><thead><tr>${source.headers.map(header => `<th scope="col">${escapeHTML(header)}</th>`).join('')}</tr></thead>
            <tbody>${source.sample_rows.map(row => `<tr>${row.map(value => `<td>${escapeHTML(value)}</td>`).join('')}</tr>`).join('')}</tbody></table>
            <p><a href="${source.source}">Source dataset</a> · ${escapeHTML(source.license)}</p>
            <p class="hash">SHA-256 ${source.sha256}</p>
          </div>
        </details>
      </div>`;
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.foundationDataset === key)));
  }

  buttons.forEach(button => button.addEventListener('click', () => render(button.dataset.foundationDataset)));
  panel.addEventListener('click', async event => {
    const button = event.target.closest('[data-copy-run]');
    if (!button) return;
    try {
      await navigator.clipboard.writeText(code);
      button.nextElementSibling.textContent = 'Copied';
    } catch {
      button.nextElementSibling.textContent = 'Select the code above or download the script.';
    }
  });
  render('jobchange');
})();
