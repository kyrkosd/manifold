/* File upload and drag-and-drop handling for the FMAS import interface. */
/* global App, API */
const Upload = {
  _dropzone: null,
  _fileInput: null,
  _fileInfo: null,

  init() {
    this._dropzone  = document.getElementById('dropzone');
    this._fileInput = document.getElementById('file-input');
    this._fileInfo  = document.getElementById('file-info');

    this._dropzone.addEventListener('dragover',  e => { e.preventDefault(); this._dropzone.classList.add('drag-over'); });
    this._dropzone.addEventListener('dragenter', e => { e.preventDefault(); this._dropzone.classList.add('drag-over'); });
    this._dropzone.addEventListener('dragleave', () => this._dropzone.classList.remove('drag-over'));
    this._dropzone.addEventListener('drop',      e => { e.preventDefault(); this._dropzone.classList.remove('drag-over'); const f = e.dataTransfer.files[0]; if (f) this.handleFile(f); });

    // Click on dropzone triggers the hidden input (unless the click is on the input itself).
    this._dropzone.addEventListener('click', e => { if (e.target !== this._fileInput) this._fileInput.click(); });
    this._dropzone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') this._fileInput.click(); });

    this._fileInput.addEventListener('change', () => {
      const f = this._fileInput.files[0];
      if (f) this.handleFile(f);
    });

    document.getElementById('change-file').addEventListener('click', () => {
      this._showDropzone();
      this._fileInput.value = '';
    });
  },

  async handleFile(file) {
    const allowed = ['.csv', '.tsv', '.xlsx', '.xls'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowed.includes(ext)) {
      App.showError(`Unsupported file type (${ext}). Please upload a CSV, TSV, or Excel file.`);
      return;
    }
    if (file.size > 500 * 1024 * 1024) {
      App.showError('File is too large. Maximum size is 500 MB.');
      return;
    }

    this._showFileInfo(file);
    App.showLoading('Uploading and analyzing…');

    try {
      const preview = await API.uploadFile(file);
      App.dataId = preview.data_id;
      App.previewData = preview;
      App.hideLoading();
      App.goToStep(1);
    } catch (err) {
      App.hideLoading();
      this._showDropzone();
      App.showError(err.error || 'Could not parse file. Make sure it is a valid CSV, TSV, or Excel file.');
    }
  },

  _showFileInfo(file) {
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = this.formatFileSize(file.size);
    this._dropzone.hidden = true;
    this._fileInfo.hidden = false;
  },

  _showDropzone() {
    this._dropzone.hidden = false;
    this._fileInfo.hidden = true;
  },

  formatFileSize(bytes) {
    if (bytes >= 1_048_576) return (bytes / 1_048_576).toFixed(1) + ' MB';
    if (bytes >= 1024)     return (bytes / 1024).toFixed(0) + ' KB';
    return bytes + ' B';
  },
};
window.Upload = Upload;
