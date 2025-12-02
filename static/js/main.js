let currentMode = 'point'; // 'point' or 'box'
let currentDirectory = null;
let currentImageName = null;
let currentImagePath = null;
let imageScale = 1;
let points = []; // Array of {x, y, label}
let boxStart = null;
let boxEnd = null;
let isDrawingBox = false;

// New State
let imageList = [];
let sortOrder = 'asc'; // 'asc' or 'desc'
let imageMasks = {}; // { imagePath: maskDataUrl }
let imagePoints = {}; // { imagePath: points } - Optional: restore points too?
// Let's just persist mask for now as requested "save segmentation result"

// Shortcuts State
let shortcuts = {
    prev: 'ArrowLeft',
    next: 'ArrowRight'
};

let saveMode = 'crop'; // 'crop' or 'mask'

const imageCanvas = document.getElementById('imageCanvas');
const maskCanvas = document.getElementById('maskCanvas');
const interactionCanvas = document.getElementById('interactionCanvas');
const ctxImage = imageCanvas.getContext('2d');
const ctxMask = maskCanvas.getContext('2d');
const ctxInteraction = interactionCanvas.getContext('2d');

const loading = document.getElementById('loading');
const resultModal = document.getElementById('resultModal');
const resultImage = document.getElementById('resultImage');
const downloadLink = document.getElementById('downloadLink');
const modelSelect = document.getElementById('modelSelect');
const fileList = document.getElementById('fileList');
const btnSort = document.getElementById('btnSort');
const btnCrop = document.getElementById('btnCrop');

// Settings Elements
const settingsModal = document.getElementById('settingsModal');
const btnSettings = document.getElementById('btnSettings');
const shortcutPrevInput = document.getElementById('shortcutPrev');
const shortcutNextInput = document.getElementById('shortcutNext');
const btnSaveSettings = document.getElementById('btnSaveSettings');

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    fetchModels();
    loadShortcuts();
});
modelSelect.addEventListener('change', handleModelChange);
document.getElementById('btnLoadDir').addEventListener('click', loadDirectory);
document.getElementById('interactionCanvas').addEventListener('mousedown', handleMouseDown);
document.getElementById('interactionCanvas').addEventListener('mousemove', handleMouseMove);
document.getElementById('interactionCanvas').addEventListener('mouseup', handleMouseUp);
document.getElementById('interactionCanvas').addEventListener('contextmenu', (e) => e.preventDefault()); // Prevent context menu for right click
document.getElementById('closeResultModal').addEventListener('click', () => resultModal.classList.add('hidden'));
document.getElementById('closeSettingsModal').addEventListener('click', () => settingsModal.classList.add('hidden'));
btnCrop.addEventListener('click', handleCrop);
document.getElementById('btnSaveAll').addEventListener('click', handleBatchSave);
document.getElementById('btnUndo').addEventListener('click', handleUndo);
document.getElementById('btnClear').addEventListener('click', clearAll);
btnSort.addEventListener('click', toggleSort);
btnSettings.addEventListener('click', openSettings);
btnSaveSettings.addEventListener('click', saveSettings);

// Save Mode Listeners
document.querySelectorAll('input[name="saveMode"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        saveMode = e.target.value;
        if (saveMode === 'crop') {
            btnCrop.textContent = 'Save Crop';
        } else if (saveMode === 'mask') {
            btnCrop.textContent = 'Save Mask';
        } else if (saveMode === 'bbox') {
            btnCrop.textContent = 'Save BBox';
        }
    });
});

// Shortcut Inputs
[shortcutPrevInput, shortcutNextInput].forEach(input => {
    input.addEventListener('keydown', (e) => {
        e.preventDefault();
        input.value = e.key;
    });
});

// Keyboard Shortcuts
document.addEventListener('keydown', (e) => {
    // Ignore if modal is open or typing in input (except shortcut inputs which handle their own)
    if (!settingsModal.classList.contains('hidden') && e.target.tagName !== 'INPUT') return;
    if (e.target.tagName === 'INPUT' && !e.target.id.startsWith('shortcut')) return;

    if (!currentDirectory || imageList.length === 0) return;

    // Ignore if typing in input
    if (e.target.tagName === 'INPUT') return;

    if (e.key === shortcuts.prev) {
        navigateImage(-1);
    } else if (e.key === shortcuts.next) {
        navigateImage(1);
    }
});

function loadShortcuts() {
    const saved = localStorage.getItem('sam_shortcuts');
    if (saved) {
        shortcuts = JSON.parse(saved);
    }
}

function openSettings() {
    shortcutPrevInput.value = shortcuts.prev;
    shortcutNextInput.value = shortcuts.next;
    settingsModal.classList.remove('hidden');
}

function saveSettings() {
    const prev = shortcutPrevInput.value;
    const next = shortcutNextInput.value;

    if (prev && next) {
        shortcuts = { prev, next };
        localStorage.setItem('sam_shortcuts', JSON.stringify(shortcuts));
        settingsModal.classList.add('hidden');
    } else {
        alert('Please set both shortcuts.');
    }
}

function navigateImage(direction) {
    if (!currentImageName) return;

    const currentIndex = imageList.indexOf(currentImageName);
    if (currentIndex === -1) return;

    let newIndex = currentIndex + direction;
    if (newIndex < 0) newIndex = 0;
    if (newIndex >= imageList.length) newIndex = imageList.length - 1;

    if (newIndex !== currentIndex) {
        const newImage = imageList[newIndex];
        const li = fileList.children[newIndex];
        selectImage(newImage, li);
        li.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
}

function toggleSort() {
    sortOrder = sortOrder === 'asc' ? 'desc' : 'asc';
    btnSort.textContent = sortOrder === 'asc' ? '⬇️' : '⬆️';
    renderFileList(imageList);
}

// Mode Switching
function setMode(mode) {
    currentMode = mode;
    document.getElementById('btnPoint').classList.toggle('active', mode === 'point');
    document.getElementById('btnBox').classList.toggle('active', mode === 'box');
    clearInteractions();
}

function handleUndo() {
    if (points.length > 0) {
        points.pop();
        drawInteractions();
        runSegmentation();
    } else if (boxStart || boxEnd) {
        boxStart = null;
        boxEnd = null;
        isDrawingBox = false;
        drawInteractions();
        runSegmentation(); // This will clear the mask if no points left
    }
}

function clearAll() {
    points = [];
    boxStart = null;
    boxEnd = null;
    currentRawMask = null;

    // Clear persisted mask
    if (currentImagePath) {
        delete imageMasks[currentImagePath];
    }

    ctxMask.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    drawInteractions();
}

function clearInteractions() {
    points = [];
    boxStart = null;
    boxEnd = null;
    drawInteractions();
}

// Directory Loading
async function loadDirectory() {
    const path = document.getElementById('dirInput').value;
    if (!path) return;

    currentDirectory = path;
    loading.classList.remove('hidden');

    try {
        const response = await fetch('/api/list_images', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: path })
        });
        const data = await response.json();

        if (data.images) {
            imageList = data.images; // Store original list
            renderFileList(imageList);
        } else {
            alert('Error loading directory: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error loading directory');
    } finally {
        loading.classList.add('hidden');
    }
}

function renderFileList(images) {
    // Sort
    const sortedImages = [...images].sort((a, b) => {
        return sortOrder === 'asc' ? a.localeCompare(b) : b.localeCompare(a);
    });

    // Update global list reference to match sorted order for navigation
    imageList = sortedImages;

    fileList.innerHTML = '';
    sortedImages.forEach(img => {
        const li = document.createElement('li');
        li.textContent = img;
        li.onclick = () => selectImage(img, li);
        if (img === currentImageName) li.classList.add('active');
        fileList.appendChild(li);
    });
}

function selectImage(filename, liElement) {
    // Update UI
    const items = fileList.querySelectorAll('li');
    items.forEach(i => i.classList.remove('active'));
    liElement.classList.add('active');

    currentImageName = filename;

    // Construct path
    if (currentDirectory.endsWith('/')) {
        currentImagePath = currentDirectory + filename;
    } else {
        currentImagePath = currentDirectory + '/' + filename;
    }

    loadImage(`/api/image?path=${encodeURIComponent(currentImagePath)}`);
}

function loadImage(url) {
    const img = new Image();
    img.onload = () => {
        // Resize canvases
        const container = document.getElementById('canvasContainer');
        const maxWidth = container.clientWidth;
        const maxHeight = container.clientHeight;

        let width = img.width;
        let height = img.height;

        // Scale down if too big
        const scaleX = maxWidth / width;
        const scaleY = maxHeight / height;
        const scale = Math.min(scaleX, scaleY); // Allow scaling up or down to fit

        imageScale = scale;

        width = width * scale;
        height = height * scale;

        [imageCanvas, maskCanvas, interactionCanvas].forEach(canvas => {
            canvas.width = width;
            canvas.height = height;
            canvas.style.width = `${width}px`;
            canvas.style.height = `${height}px`;
        });

        ctxImage.drawImage(img, 0, 0, width, height);
        document.querySelector('.placeholder').style.display = 'none';

        // Clear interactions but check for persisted mask
        clearInteractions();
        ctxMask.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

        if (currentImagePath && imageMasks[currentImagePath]) {
            currentRawMask = imageMasks[currentImagePath];
            drawMask(currentRawMask);
        } else {
            currentRawMask = null;
        }
    };
    img.src = url;
}

// Interaction Handling
function handleMouseDown(e) {
    if (!currentImagePath) return;

    const rect = interactionCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left);
    const y = (e.clientY - rect.top);

    if (currentMode === 'point') {
        // Left click = positive (1), Right click = negative (0)
        const label = e.button === 2 ? 0 : 1;
        points.push({ x, y, label });
        drawInteractions();
        runSegmentation();
    } else if (currentMode === 'box') {
        isDrawingBox = true;
        boxStart = { x, y };
        boxEnd = { x, y };
        // Clear previous box if any, or maybe we want to support multiple? 
        // For now, single box.
    }
}

function handleMouseMove(e) {
    if (!currentImagePath || !isDrawingBox) return;

    const rect = interactionCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left);
    const y = (e.clientY - rect.top);

    boxEnd = { x, y };
    drawInteractions();
}

function handleMouseUp(e) {
    if (!currentImagePath || !isDrawingBox) return;

    isDrawingBox = false;
    runSegmentation();
}

function drawInteractions() {
    ctxInteraction.clearRect(0, 0, interactionCanvas.width, interactionCanvas.height);

    // Draw Points
    points.forEach(p => {
        ctxInteraction.beginPath();
        ctxInteraction.arc(p.x, p.y, 5, 0, 2 * Math.PI);
        ctxInteraction.fillStyle = p.label === 1 ? '#10b981' : '#ef4444'; // Green for pos, Red for neg
        ctxInteraction.fill();
        ctxInteraction.strokeStyle = 'white';
        ctxInteraction.lineWidth = 2;
        ctxInteraction.stroke();
    });

    // Draw Box
    if (boxStart && boxEnd) {
        const x = Math.min(boxStart.x, boxEnd.x);
        const y = Math.min(boxStart.y, boxEnd.y);
        const w = Math.abs(boxEnd.x - boxStart.x);
        const h = Math.abs(boxEnd.y - boxStart.y);

        ctxInteraction.strokeStyle = '#3b82f6';
        ctxInteraction.lineWidth = 2;
        ctxInteraction.strokeRect(x, y, w, h);
    }
}

async function runSegmentation() {
    if (!currentImagePath) return;
    if (points.length === 0 && (!boxStart || !boxEnd)) return;

    // Prepare data
    // Scale coordinates back to original image size
    const scaledPoints = points.map(p => [p.x / imageScale, p.y / imageScale]);
    const scaledLabels = points.map(p => p.label);

    let scaledBox = null;
    if (boxStart && boxEnd) {
        const x1 = Math.min(boxStart.x, boxEnd.x) / imageScale;
        const y1 = Math.min(boxStart.y, boxEnd.y) / imageScale;
        const x2 = Math.max(boxStart.x, boxEnd.x) / imageScale;
        const y2 = Math.max(boxStart.y, boxEnd.y) / imageScale;

        // Avoid zero area box
        if (Math.abs(x2 - x1) > 1 && Math.abs(y2 - y1) > 1) {
            scaledBox = [x1, y1, x2, y2];
        }
    }

    // Don't run if we only have a tiny box (accidental click in box mode)
    if (currentMode === 'box' && !scaledBox) return;

    loading.classList.remove('hidden');

    try {
        const response = await fetch('/api/segment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_path: currentImagePath,
                points: scaledPoints.length > 0 ? scaledPoints : null,
                labels: scaledLabels.length > 0 ? scaledLabels : null,
                box: scaledBox
            })
        });

        const data = await response.json();

        if (data.mask) {
            drawMask(data.mask);
        } else if (data.error) {
            console.error(data.error);
        }
    } catch (error) {
        console.error('Segmentation failed:', error);
    } finally {
        loading.classList.add('hidden');
    }
}

function drawMask(maskDataUrl) {
    const img = new Image();
    img.onload = () => {
        ctxMask.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        // Draw mask with color
        // The mask image is white on black (or similar). We want to draw it as a colored overlay.
        // But here we receive a base64 PNG. Let's draw it to an offscreen canvas to manipulate pixels if needed,
        // or just draw it with globalCompositeOperation if it's already transparent?
        // The backend returns a grayscale or binary mask usually.
        // Let's assume backend returns a white mask on black background or transparent.
        // Actually, my backend code returns a grayscale image (0-255).

        // To make it a nice overlay:
        // 1. Draw mask to temp canvas
        // 2. Get image data
        // 3. Set alpha based on pixel value
        // 4. Set color (e.g. blue)

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = maskCanvas.width;
        tempCanvas.height = maskCanvas.height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(img, 0, 0, maskCanvas.width, maskCanvas.height);

        const imageData = tempCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
        const data = imageData.data;

        for (let i = 0; i < data.length; i += 4) {
            const val = data[i]; // R channel (grayscale so R=G=B)
            if (val > 0) {
                data[i] = 59;     // R
                data[i + 1] = 130;// G
                data[i + 2] = 246;// B (Blue-ish)
                data[i + 3] = 150;// Alpha
            } else {
                data[i + 3] = 0;  // Transparent
            }
        }

        ctxMask.putImageData(imageData, 0, 0);
    };
    img.src = maskDataUrl;
}

async function handleCrop() {
    if (!currentFilename) return;

    // We need the current mask. 
    // We can ask the backend to crop based on the last state, 
    // OR we can send the mask back? 
    // Sending mask back is safer if we want exactly what is seen.
    // But the mask on canvas is resized.
    // Better to re-run segmentation or cache the last mask on backend?
    // Or just send the current points/box again to 'crop' endpoint?
    // Or, since we have the mask as a data URL from the last 'segment' call, we could store that.

    // Let's modify runSegmentation to store the raw mask data url
    // Actually, let's just grab the mask from the maskCanvas? No, that's low res/colored.

    // Simplest: The backend 'segment' endpoint returned a mask. We should have saved it.
    // Let's save it in a variable.
}

// We need to store the raw mask from the server
let currentRawMask = null;

// Override drawMask to store it
const originalDrawMask = drawMask;
drawMask = function (maskDataUrl) {
    currentRawMask = maskDataUrl;
    if (currentImagePath) {
        imageMasks[currentImagePath] = maskDataUrl;
    }
    originalDrawMask(maskDataUrl);
}

// Redefine handleCrop
async function handleCrop() {
    if (!currentImagePath || !currentRawMask) {
        alert("Please select an object first.");
        return;
    }

    const label = document.getElementById('classLabel').value;

    loading.classList.remove('hidden');

    try {
        const response = await fetch('/api/crop', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_path: currentImagePath,
                mask: currentRawMask,
                save_mode: saveMode,
                label: label
            })
        });
        const data = await response.json();

        if (saveMode === 'bbox') {
            // BBox mode: no image to display
            if (data.message) {
                alert(data.message + '\nSaved to: ' + data.saved_path);
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        } else if (data.cropped_image) {
            resultImage.src = data.cropped_image;
            downloadLink.href = data.cropped_image;
            downloadLink.download = saveMode === 'crop' ? 'crop.png' : 'mask.png';
            resultModal.classList.remove('hidden');

            if (data.saved_path) {
                console.log('Saved to:', data.saved_path);
            }
        } else {
            alert('Error: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error processing request');
    } finally {
        loading.classList.add('hidden');
    }
}

async function handleBatchSave() {
    const items = Object.entries(imageMasks).map(([path, mask]) => ({
        image_path: path,
        mask: mask
    }));

    if (items.length === 0) {
        alert("No segmented images to save.");
        return;
    }

    const label = document.getElementById('classLabel').value;

    loading.classList.remove('hidden');

    try {
        const response = await fetch('/api/batch_crop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                items: items,
                save_mode: saveMode,
                label: label
            })
        });
        const data = await response.json();

        if (data.saved) {
            const itemType = saveMode === 'crop' ? 'crops' : (saveMode === 'mask' ? 'masks' : 'bboxes');
            let msg = `Successfully saved ${data.saved.length} ${itemType}.`;
            if (data.errors && data.errors.length > 0) {
                msg += `\nErrors: ${data.errors.length}`;
            }
            alert(msg);
        } else {
            alert('Batch save failed');
        }
    } catch (error) {
        console.error('Batch save error:', error);
        alert('Batch save error');
    } finally {
        loading.classList.add('hidden');
    }
}

async function fetchModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();

        modelSelect.innerHTML = '';
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            if (model === data.current) {
                option.selected = true;
            }
            modelSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error fetching models:', error);
    }
}

async function handleModelChange(e) {
    const modelName = e.target.value;
    loading.classList.remove('hidden');
    try {
        const response = await fetch('/api/set_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: modelName })
        });
        const data = await response.json();
        if (data.status !== 'success') {
            alert('Failed to switch model');
        }
    } catch (error) {
        console.error('Error switching model:', error);
        alert('Error switching model');
    } finally {
        loading.classList.add('hidden');
    }
}
