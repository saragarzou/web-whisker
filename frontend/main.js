const API_BASE = 'https://web-whisker.onrender.com';
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
let isDrawing = false;
let x = 0, y = 0;
const ongoingTouches = [];

context.lineCap = 'round';
context.lineJoin = 'round';
context.lineWidth = 28;
context.strokeStyle = 'white';

function getTouchPos(touch) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: touch.clientX - rect.left,
    y: touch.clientY - rect.top
  };
}

function copyTouch(touch) {
  return { identifier: touch.identifier, ...getTouchPos(touch) };
}

function ongoingTouchIndexById(idToFind) {
  return ongoingTouches.findIndex(t => t.identifier === idToFind);
}

function drawLine(x1, y1, x2, y2) {
  context.beginPath();
  context.moveTo(x1, y1);
  context.lineTo(x2, y2);
  context.stroke();
}

// Mouse events
canvas.addEventListener('mousedown', (e) => {
  x = e.offsetX;
  y = e.offsetY;
  isDrawing = true;
  context.beginPath();
  context.moveTo(x, y);
});

canvas.addEventListener('mousemove', (e) => {
  if (isDrawing) {
    drawLine(x, y, e.offsetX, e.offsetY);
    x = e.offsetX;
    y = e.offsetY;
  }
});

canvas.addEventListener('mouseup', () => {
  isDrawing = false;
});

// Touch events
canvas.addEventListener('touchstart', (e) => {
  e.preventDefault();
  for (let touch of e.changedTouches) {
    const pos = getTouchPos(touch);
    context.beginPath();
    context.moveTo(pos.x, pos.y);
    ongoingTouches.push(copyTouch(touch));
  }
});

canvas.addEventListener('touchmove', (e) => {
  e.preventDefault();
  for (let touch of e.changedTouches) {
    const idx = ongoingTouchIndexById(touch.identifier);
    if (idx >= 0) {
      const oldTouch = ongoingTouches[idx];
      const newPos = getTouchPos(touch);
      drawLine(oldTouch.x, oldTouch.y, newPos.x, newPos.y);
      ongoingTouches[idx] = { identifier: touch.identifier, ...newPos };
    }
  }
});

canvas.addEventListener('touchend', (e) => {
  e.preventDefault();
  for (let touch of e.changedTouches) {
    const idx = ongoingTouchIndexById(touch.identifier);
    if (idx >= 0) ongoingTouches.splice(idx, 1);
  }
});

canvas.addEventListener('touchcancel', (e) => {
  e.preventDefault();
  for (let touch of e.changedTouches) {
    const idx = ongoingTouchIndexById(touch.identifier);
    if (idx >= 0) ongoingTouches.splice(idx, 1);
  }
});

function clearCanvas() {
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = 'black';
  context.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById('result').textContent = '';
}

function submitDrawing() {
  const dataURL = canvas.toDataURL('image/png');
  const resultEl = document.getElementById('result');
  resultEl.textContent = 'predicting...';

  fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ image: dataURL })
  })
  .then(response => response.json())
  .then(data => {
    const prediction = data.prediction;
    const words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
    resultEl.innerHTML = '';

    const label = document.createElement('div');
    label.textContent = `prediction: ${words[prediction]} cat${prediction !== 1 ? 's' : ''}`;
    label.style.fontWeight = 'bold';
    resultEl.appendChild(label);

    const cats = document.createElement('div');
    for (let i = 0; i < prediction; i++) {
      const cat = document.createElement('span');
      cat.textContent = '₍^. .^₎⟆';
      cat.className = 'cat';
      cats.appendChild(cat);
    }
    resultEl.appendChild(cats);
  })
  .catch(err => {
    console.error(err);
    resultEl.textContent = 'prediction failed';
  });
}

window.onload = clearCanvas;
