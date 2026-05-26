/* Main app state and step navigation for the FMAS import interface. */
const App = {
  currentStep: 0,
  dataId: null,
  previewData: null,
  config: {
    n_charts: 'auto',
    threshold_method: 'adaptive',
    normalization: 'standard',
    max_iterations: 10,
    overlap_factor: 0.2,
    band_method: 'spectral_gaps',
  },

  init() {
    // Tab switching between File and SQL panels.
    document.getElementById('tab-file').addEventListener('click', () => this._switchTab('file'));
    document.getElementById('tab-sql').addEventListener('click', () => this._switchTab('sql'));

    // Step pill navigation.
    document.querySelectorAll('.step-pill').forEach(pill => {
      pill.addEventListener('click', () => this.goToStep(+pill.dataset.step));
      pill.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') this.goToStep(+pill.dataset.step); });
    });

    // Navigation buttons.
    document.getElementById('btn-next-1').addEventListener('click', () => this.goToStep(1));
    document.getElementById('btn-back-2').addEventListener('click', () => this.goToStep(0));
    document.getElementById('btn-next-2').addEventListener('click', () => this.goToStep(2));
    document.getElementById('btn-back-3').addEventListener('click', () => this.goToStep(1));

    // Error toast dismiss.
    document.getElementById('error-dismiss').addEventListener('click', () => this.hideError());

    Upload.init();
    SQL.init();
    Preview.init();
    Config.init();

    this._updateStepUI(0);
  },

  // ------------------------------------------------------------------

  goToStep(n) {
    if (n === 1 && !this.dataId) { this.showError('Load a dataset first.'); return; }
    if (n === 2 && !this.previewData) { this.showError('Complete the preview step first.'); return; }
    this._fadeOutIn(n);
  },

  _fadeOutIn(targetStep) {
    const panes = ['pane-source', 'pane-preview', 'pane-config'];
    const current = document.getElementById(panes[this.currentStep]);
    const next    = document.getElementById(panes[targetStep]);

    current.classList.add('pane-exit');
    setTimeout(() => {
      current.hidden = true;
      current.classList.remove('pane-exit');
      next.hidden = false;
      next.classList.add('pane-enter');
      requestAnimationFrame(() => {
        requestAnimationFrame(() => next.classList.remove('pane-enter'));
      });
      this.currentStep = targetStep;
      this._updateStepUI(targetStep);
      if (targetStep === 1 && this.previewData) Preview.render(this.previewData);
      if (targetStep === 2) Config.updateSummary();
    }, 150);
  },

  _updateStepUI(step) {
    document.getElementById('btn-next-1').disabled = !this.dataId;
    document.querySelectorAll('.step-pill').forEach((pill, i) => {
      pill.classList.toggle('active', i === step);
      pill.classList.toggle('done',   i < step);
      pill.setAttribute('aria-current', i === step ? 'step' : 'false');
    });
  },

  _switchTab(tab) {
    const isFile = tab === 'file';
    document.getElementById('tab-file').classList.toggle('active', isFile);
    document.getElementById('tab-sql').classList.toggle('active', !isFile);
    document.getElementById('tab-file').setAttribute('aria-selected', isFile);
    document.getElementById('tab-sql').setAttribute('aria-selected', !isFile);
    document.getElementById('area-file').hidden = !isFile;
    document.getElementById('area-sql').hidden  =  isFile;
  },

  // ------------------------------------------------------------------

  showLoading(message = 'Processing…') {
    document.getElementById('loading-message').textContent = message;
    document.getElementById('loading-overlay').hidden = false;
  },
  hideLoading() { document.getElementById('loading-overlay').hidden = true; },

  _errorTimer: null,
  showError(message) {
    document.getElementById('error-message').textContent = message;
    document.getElementById('error-toast').hidden = false;
    clearTimeout(this._errorTimer);
    this._errorTimer = setTimeout(() => this.hideError(), 8000);
  },
  hideError() { document.getElementById('error-toast').hidden = true; },
};

document.addEventListener('DOMContentLoaded', () => App.init());
