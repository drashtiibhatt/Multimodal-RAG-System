// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global state
let selectedFile = null;

// ============================================================================
// Utility Functions
// ============================================================================

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');

    const colors = {
        success: 'bg-green-500',
        error: 'bg-red-500',
        info: 'bg-blue-500',
        warning: 'bg-yellow-500'
    };

    toast.className = `${colors[type]} text-white px-6 py-3 rounded-lg shadow-lg fade-in`;
    toast.textContent = message;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => container.removeChild(toast), 300);
    }, 3000);
}

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active', 'border-blue-500', 'text-blue-600');
        button.classList.add('border-transparent', 'text-gray-500', 'hover:text-gray-700', 'hover:border-gray-300');
    });

    document.getElementById(`tab-${tabName}`).classList.add('active', 'border-blue-500', 'text-blue-600');
    document.getElementById(`tab-${tabName}`).classList.remove('border-transparent', 'text-gray-500');

    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.add('hidden');
    });

    document.getElementById(`content-${tabName}`).classList.remove('hidden');

    // Load data for specific tabs
    if (tabName === 'documents') {
        loadDocuments();
    } else if (tabName === 'settings') {
        loadSettings();
    } else if (tabName === 'stats') {
        loadStatistics();
    }
}

// ============================================================================
// Query Functions
// ============================================================================

async function executeQuery() {
    const queryInput = document.getElementById('query-input').value.trim();

    if (!queryInput) {
        showToast('Please enter a query', 'warning');
        return;
    }

    const button = document.getElementById('query-button');
    const buttonText = document.getElementById('query-button-text');

    // Disable button and show loading
    button.disabled = true;
    buttonText.innerHTML = '<div class="spinner mx-auto"></div>';

    const requestData = {
        query: queryInput,
        top_k: parseInt(document.getElementById('top-k').value),
        use_hybrid: document.getElementById('use-hybrid').checked,
        use_guards: document.getElementById('use-guards').checked,
        min_confidence: parseFloat(document.getElementById('min-confidence').value)
    };

    try {
        const response = await fetch(`${API_BASE_URL}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Query failed');
        }

        const data = await response.json();
        displayResults(data);
        showToast('✓ Query completed successfully', 'success');

    } catch (error) {
        showToast('Error: ' + error.message, 'error');
        console.error('Query error:', error);

        document.getElementById('results-container').innerHTML = `
            <div class="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">
                <strong>Error:</strong> ${error.message}
            </div>
        `;
    } finally {
        button.disabled = false;
        buttonText.textContent = '🚀 Generate Use Cases';
    }
}

function displayResults(data) {
    const container = document.getElementById('results-container');

    if (!data.use_cases || data.use_cases.length === 0) {
        container.innerHTML = `
            <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-yellow-800">
                <strong>No use cases generated.</strong><br>
                ${data.metadata.missing_information ? data.metadata.missing_information.join('<br>') : ''}
            </div>
        `;
        return;
    }

    let html = '';

    // Metadata summary
    html += `
        <div class="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
            <div class="grid grid-cols-2 gap-4 text-sm">
                <div>
                    <span class="text-gray-600">Query:</span>
                    <span class="font-semibold ml-2">${data.metadata.query}</span>
                </div>
                <div>
                    <span class="text-gray-600">Results Found:</span>
                    <span class="font-semibold ml-2">${data.metadata.results_found}</span>
                </div>
                <div>
                    <span class="text-gray-600">Confidence:</span>
                    <span class="font-semibold ml-2">${(data.metadata.confidence_score * 100).toFixed(1)}%</span>
                </div>
                <div>
                    <span class="text-gray-600">Use Cases:</span>
                    <span class="font-semibold ml-2">${data.use_cases.length}</span>
                </div>
            </div>
        </div>
    `;

    // Safety report
    if (data.safety_report) {
        const report = data.safety_report;
        const statusColor = report.passed ? 'green' : 'red';

        html += `
            <div class="bg-${statusColor}-50 border border-${statusColor}-200 rounded-lg p-4 mb-4">
                <h4 class="font-semibold text-${statusColor}-900 mb-2">
                    ${report.passed ? '✓' : '⚠'} Safety Report
                </h4>
                <div class="text-sm space-y-1">
                    <div>Checks: ${report.checks_passed}/${report.total_checks_run} passed</div>
                    ${report.pii_detected ? `<div class="text-${statusColor}-700">⚠ PII Detected: ${report.pii_types_found.join(', ')}</div>` : ''}
                    ${report.warnings.length > 0 ? `<div class="text-yellow-700">Warnings: ${report.warnings.length}</div>` : ''}
                </div>
            </div>
        `;
    }

    // Use cases
    data.use_cases.forEach((uc, index) => {
        html += `
            <div class="border border-gray-200 rounded-lg p-6 mb-4 hover:shadow-md transition">
                <div class="flex items-start justify-between mb-3">
                    <h3 class="text-lg font-semibold text-gray-900 flex-1">
                        ${index + 1}. ${uc.title}
                    </h3>
                    <span class="ml-4 px-3 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full">
                        ${uc.steps ? uc.steps.length : 0} steps
                    </span>
                </div>

                <div class="space-y-3">
                    <div>
                        <span class="font-medium text-gray-700">Goal:</span>
                        <p class="text-gray-600 mt-1">${uc.goal}</p>
                    </div>

                    ${uc.preconditions && uc.preconditions.length > 0 ? `
                        <div>
                            <span class="font-medium text-gray-700">Preconditions:</span>
                            <ul class="list-disc list-inside text-gray-600 mt-1">
                                ${uc.preconditions.map(p => `<li>${p}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}

                    <div>
                        <span class="font-medium text-gray-700">Steps:</span>
                        <ol class="list-decimal list-inside text-gray-600 mt-1 space-y-1">
                            ${uc.steps ? uc.steps.map(s => `<li>${s}</li>`).join('') : ''}
                        </ol>
                    </div>

                    ${uc.expected_results && uc.expected_results.length > 0 ? `
                        <div>
                            <span class="font-medium text-gray-700">Expected Results:</span>
                            <ul class="list-disc list-inside text-gray-600 mt-1">
                                ${uc.expected_results.map(r => `<li>${r}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}

                    ${uc.test_data ? `
                        <div>
                            <span class="font-medium text-gray-700">Test Data:</span>
                            <pre class="bg-gray-50 p-2 rounded mt-1 text-sm overflow-x-auto">${JSON.stringify(uc.test_data, null, 2)}</pre>
                        </div>
                    ` : ''}

                    <div class="grid grid-cols-2 gap-4 pt-3 border-t">
                        ${uc.negative_cases && uc.negative_cases.length > 0 ? `
                            <div>
                                <span class="text-sm font-medium text-gray-700">Negative Cases:</span>
                                <ul class="list-disc list-inside text-sm text-gray-600 mt-1">
                                    ${uc.negative_cases.map(n => `<li>${n}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}

                        ${uc.boundary_cases && uc.boundary_cases.length > 0 ? `
                            <div>
                                <span class="text-sm font-medium text-gray-700">Boundary Cases:</span>
                                <ul class="list-disc list-inside text-sm text-gray-600 mt-1">
                                    ${uc.boundary_cases.map(b => `<li>${b}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

// ============================================================================
// Document Management Functions
// ============================================================================

// Helper function to get file type icon
function getFileIcon(filename) {
    const ext = filename.toLowerCase().split('.').pop();
    const imageExts = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'];

    if (imageExts.includes(ext)) {
        return `<svg class="w-10 h-10 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
        </svg>`;
    } else if (ext === 'pdf') {
        return `<svg class="w-10 h-10 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"></path>
        </svg>`;
    } else if (ext === 'docx') {
        return `<svg class="w-10 h-10 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
        </svg>`;
    } else {
        return `<svg class="w-10 h-10 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
        </svg>`;
    }
}

function isImageFile(filename) {
    const ext = filename.toLowerCase().split('.').pop();
    return ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'].includes(ext);
}

function handleFileSelect(event) {
    selectedFile = event.target.files[0];

    if (selectedFile) {
        displayFilePreview(selectedFile);
    }
}

function displayFilePreview(file) {
    const uploadButton = document.getElementById('upload-button');
    const previewDiv = document.getElementById('file-preview');
    const filenameEl = document.getElementById('preview-filename');
    const detailsEl = document.getElementById('preview-details');
    const iconEl = document.getElementById('preview-icon');
    const imageThumbnail = document.getElementById('image-thumbnail');
    const thumbnailImg = document.getElementById('thumbnail-img');

    // Enable upload button
    uploadButton.disabled = false;

    // Show preview
    previewDiv.classList.remove('hidden');

    // Set filename and details
    filenameEl.textContent = file.name;
    detailsEl.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB • ${file.type || 'Unknown type'}`;

    // Set icon
    iconEl.innerHTML = getFileIcon(file.name);

    // Show image thumbnail if it's an image
    if (isImageFile(file.name)) {
        const reader = new FileReader();
        reader.onload = function(e) {
            thumbnailImg.src = e.target.result;
            imageThumbnail.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    } else {
        imageThumbnail.classList.add('hidden');
    }
}

function clearFileSelection() {
    selectedFile = null;
    document.getElementById('file-input').value = '';
    document.getElementById('file-preview').classList.add('hidden');
    document.getElementById('upload-button').disabled = true;
}

async function uploadDocument() {
    if (!selectedFile) {
        showToast('Please select a file first', 'warning');
        return;
    }

    const uploadButton = document.getElementById('upload-button');
    const statusDiv = document.getElementById('upload-status');
    const isImage = isImageFile(selectedFile.name);

    uploadButton.disabled = true;
    uploadButton.innerHTML = `
        <div class="flex items-center justify-center">
            <div class="spinner mr-2" style="width: 20px; height: 20px;"></div>
            ${isImage ? 'Processing Image (OCR/Vision)...' : 'Uploading...'}
        </div>
    `;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const result = await response.json();

        statusDiv.className = 'bg-green-50 border border-green-200 rounded-lg p-3 text-sm text-green-800';
        statusDiv.innerHTML = `
            <strong>✓ Success!</strong><br>
            ${isImage ? 'Image processed with OCR/Vision API' : 'Document indexed'}: ${result.total_vectors} total vectors
        `;
        statusDiv.classList.remove('hidden');

        showToast(isImage ? 'Image uploaded & processed!' : 'Document uploaded successfully!', 'success');

        // Clear selection
        clearFileSelection();

        // Refresh documents list
        loadDocuments();
        updateQuickStats();

    } catch (error) {
        statusDiv.className = 'bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-800';
        statusDiv.innerHTML = `<strong>✗ Error:</strong> ${error.message}`;
        statusDiv.classList.remove('hidden');
        showToast('Upload failed: ' + error.message, 'error');
    } finally {
        uploadButton.disabled = false;
        uploadButton.textContent = '📤 Upload & Index';
    }
}

async function loadDocuments() {
    const container = document.getElementById('documents-list');
    container.innerHTML = '<div class="text-center text-gray-500 py-8"><div class="spinner mx-auto"></div><p class="mt-2">Loading...</p></div>';

    try {
        const response = await fetch(`${API_BASE_URL}/api/documents`);

        if (!response.ok) throw new Error('Failed to load documents');

        const documents = await response.json();

        if (documents.length === 0) {
            container.innerHTML = `
                <div class="text-center text-gray-500 py-8">
                    <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <p class="mt-2">No documents indexed yet</p>
                </div>
            `;
            return;
        }

        let html = '<div class="space-y-3">';

        documents.forEach(doc => {
            const fileIcon = getFileIcon(doc.source);
            const isImage = isImageFile(doc.source);
            const fileType = isImage ? '🖼️ Image' : '📄 Document';

            html += `
                <div class="border border-gray-200 rounded-lg p-4 hover:shadow-md transition">
                    <div class="flex items-start justify-between">
                        <div class="flex items-start space-x-3 flex-1">
                            <div class="flex-shrink-0">
                                ${fileIcon}
                            </div>
                            <div class="flex-1 min-w-0">
                                <h4 class="font-medium text-gray-900 truncate">${doc.source}</h4>
                                <p class="text-sm text-gray-600 mt-1">
                                    ${fileType} • ${doc.total_chunks} chunks indexed
                                </p>
                            </div>
                        </div>
                        <button
                            onclick="deleteDocument('${encodeURIComponent(doc.source)}')"
                            class="ml-4 text-red-600 hover:text-red-700 text-sm font-medium flex-shrink-0">
                            Delete
                        </button>
                    </div>
                </div>
            `;
        });

        html += '</div>';
        container.innerHTML = html;

    } catch (error) {
        container.innerHTML = `
            <div class="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">
                Error loading documents: ${error.message}
            </div>
        `;
    }
}

async function deleteDocument(source) {
    if (!confirm('Are you sure you want to delete this document?')) return;

    try {
        const response = await fetch(`${API_BASE_URL}/api/documents/${source}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Delete failed');
        }

        showToast('Document deleted successfully', 'success');
        loadDocuments();
        updateQuickStats();

    } catch (error) {
        showToast('Delete failed: ' + error.message, 'error');
    }
}

async function clearAllDocuments() {
    if (!confirm('Are you sure you want to delete ALL documents? This cannot be undone.')) return;

    try {
        const response = await fetch(`${API_BASE_URL}/api/documents/clear`, {
            method: 'POST'
        });

        if (!response.ok) throw new Error('Clear failed');

        const result = await response.json();
        showToast(`All documents cleared (${result.vectors_deleted} vectors)`, 'success');

        loadDocuments();
        updateQuickStats();

    } catch (error) {
        showToast('Clear failed: ' + error.message, 'error');
    }
}

// ============================================================================
// Settings Functions
// ============================================================================

async function loadSettings() {
    const form = document.getElementById('settings-form');
    form.innerHTML = '<div class="text-center py-8"><div class="spinner mx-auto"></div></div>';

    try {
        const response = await fetch(`${API_BASE_URL}/api/settings`);
        if (!response.ok) throw new Error('Failed to load settings');

        const settings = await response.json();

        form.innerHTML = `
            <div class="space-y-4">
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">LLM Provider</label>
                        <select id="setting-provider" class="w-full px-3 py-2 border border-gray-300 rounded-lg">
                            <option value="openai" ${settings.llm_provider === 'openai' ? 'selected' : ''}>OpenAI</option>
                            <option value="ollama" ${settings.llm_provider === 'ollama' ? 'selected' : ''}>Ollama (Local)</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Temperature</label>
                        <input type="number" id="setting-temperature" value="${settings.temperature}" step="0.1" min="0" max="2"
                               class="w-full px-3 py-2 border border-gray-300 rounded-lg">
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">OpenAI Model</label>
                        <input type="text" id="setting-model" value="${settings.llm_model}"
                               class="w-full px-3 py-2 border border-gray-300 rounded-lg">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Ollama Model</label>
                        <input type="text" id="setting-ollama-model" value="${settings.ollama_model}"
                               class="w-full px-3 py-2 border border-gray-300 rounded-lg">
                    </div>
                </div>

                <div class="space-y-2 bg-gray-50 p-4 rounded-lg">
                    <label class="flex items-center">
                        <input type="checkbox" id="setting-vision" ${settings.use_vision_api ? 'checked' : ''} class="rounded text-blue-600 mr-2">
                        <span class="text-sm font-medium">Use Vision API for OCR</span>
                    </label>
                    <label class="flex items-center">
                        <input type="checkbox" id="setting-hybrid" ${settings.enable_hybrid_retrieval ? 'checked' : ''} class="rounded text-blue-600 mr-2">
                        <span class="text-sm font-medium">Enable Hybrid Retrieval</span>
                    </label>
                    <label class="flex items-center">
                        <input type="checkbox" id="setting-hallucination" ${settings.enable_hallucination_check ? 'checked' : ''} class="rounded text-blue-600 mr-2">
                        <span class="text-sm font-medium">Enable Hallucination Detection</span>
                    </label>
                </div>

                <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm text-yellow-800">
                    ⚠️ Note: Some settings require server restart to take effect
                </div>

                <button onclick="saveSettings()" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg">
                    💾 Save Settings
                </button>
            </div>
        `;

    } catch (error) {
        form.innerHTML = `<div class="text-red-600">Error loading settings: ${error.message}</div>`;
    }
}

async function saveSettings() {
    const updates = {
        llm_provider: document.getElementById('setting-provider').value,
        llm_model: document.getElementById('setting-model').value,
        temperature: parseFloat(document.getElementById('setting-temperature').value),
        use_vision_api: document.getElementById('setting-vision').checked,
        enable_hybrid_retrieval: document.getElementById('setting-hybrid').checked
    };

    try {
        const response = await fetch(`${API_BASE_URL}/api/settings`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updates)
        });

        if (!response.ok) throw new Error('Failed to save settings');

        showToast('Settings saved successfully', 'success');

    } catch (error) {
        showToast('Save failed: ' + error.message, 'error');
    }
}

// ============================================================================
// Statistics Functions
// ============================================================================

async function loadStatistics() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/stats`);
        if (!response.ok) throw new Error('Failed to load statistics');

        const stats = await response.json();

        // Vector store stats
        const vsContainer = document.getElementById('stats-vectorstore');
        vsContainer.innerHTML = displayStats(stats.vector_store);

        // Embedding cache stats
        const ecContainer = document.getElementById('stats-embedding-cache');
        ecContainer.innerHTML = displayStats(stats.cache.embedding_cache);

        // Query cache stats
        const qcContainer = document.getElementById('stats-query-cache');
        qcContainer.innerHTML = displayStats(stats.cache.query_cache);

    } catch (error) {
        showToast('Failed to load statistics', 'error');
    }
}

function displayStats(stats) {
    let html = '';
    for (const [key, value] of Object.entries(stats)) {
        html += `
            <div class="flex justify-between text-sm">
                <span class="text-gray-600">${key.replace(/_/g, ' ')}:</span>
                <span class="font-semibold">${value}</span>
            </div>
        `;
    }
    return html;
}

async function clearCache() {
    if (!confirm('Clear all caches?')) return;

    try {
        const response = await fetch(`${API_BASE_URL}/api/cache/clear`, {
            method: 'POST'
        });

        if (!response.ok) throw new Error('Failed to clear cache');

        const result = await response.json();
        showToast(`Cache cleared: ${result.embeddings_cleared} embeddings, ${result.queries_cleared} queries`, 'success');

        loadStatistics();

    } catch (error) {
        showToast('Clear cache failed: ' + error.message, 'error');
    }
}

// ============================================================================
// Quick Stats
// ============================================================================

async function updateQuickStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/info`);
        if (!response.ok) return;

        const info = await response.json();

        document.getElementById('quick-docs').textContent = info.vector_store?.unique_sources || '0';
        document.getElementById('quick-vectors').textContent = info.vector_store?.total_vectors || '0';
        document.getElementById('quick-provider').textContent = info.llm_provider || '-';

    } catch (error) {
        console.error('Failed to update quick stats:', error);
    }
}

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    updateQuickStats();
    setInterval(updateQuickStats, 30000); // Update every 30 seconds
});

// Allow Enter key to submit query
document.addEventListener('DOMContentLoaded', () => {
    const queryInput = document.getElementById('query-input');
    if (queryInput) {
        queryInput.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                executeQuery();
            }
        });
    }

    // Setup drag and drop for file upload
    const dropZone = document.getElementById('drop-zone');
    if (dropZone) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.add('border-blue-500', 'bg-blue-50');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.remove('border-blue-500', 'bg-blue-50');
            }, false);
        });

        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                selectedFile = files[0];
                displayFilePreview(selectedFile);
                showToast('File selected: ' + selectedFile.name, 'success');
            }
        }, false);
    }
});
