const input = document.getElementById('symptom-input');
const btn = document.getElementById('check-btn');
const resultArea = document.getElementById('result-area');
const template = document.getElementById('result-template');
const useLocationBox = document.getElementById('use-location');

function getLocation() {
  return new Promise((resolve) => {
    if (!useLocationBox.checked || !navigator.geolocation) {
      resolve({ latitude: null, longitude: null });
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ latitude: pos.coords.latitude, longitude: pos.coords.longitude }),
      () => resolve({ latitude: null, longitude: null }),
      { timeout: 4000 }
    );
  });
}

function renderError(message) {
  resultArea.classList.remove('hidden');
  const card = document.createElement('div');
  card.className = 'result-card';
  const label = document.createElement('p');
  label.className = 'card-label';
  label.textContent = 'Notice';
  const body = document.createElement('p');
  body.textContent = message;
  card.append(label, body);
  resultArea.replaceChildren(card);
}

function renderResult(data) {
  resultArea.classList.remove('hidden');
  resultArea.innerHTML = '';
  const node = template.content.cloneNode(true);

  node.querySelector('.disease-name').textContent = data.predicted_disease;
  node.querySelector('.confidence-fill').style.width = `${Math.round(data.confidence * 100)}%`;
  node.querySelector('.confidence-text').textContent =
    `${Math.round(data.confidence * 100)}% model confidence` +
    (data.matched_symptoms.length ? ` · based on: ${data.matched_symptoms.join(', ')}` : '');

  const top3El = node.querySelector('.top3-list');
  data.top3_predictions.forEach((p, i) => {
    if (i === 0) return; // already shown as the primary card
    const span = document.createElement('span');
    span.textContent = `${p.disease} (${Math.round(p.confidence * 100)}%)`;
    top3El.appendChild(span);
  });

  const badge = node.querySelector('.urgency-badge');
  badge.textContent = data.urgency;
  badge.classList.add(data.urgency);
  node.querySelector('.urgency-note').textContent = data.urgency_note;

  node.querySelector('.specialist-name').textContent = data.specialist;
  const nearbyNote = node.querySelector('.nearby-note');
  nearbyNote.textContent = data.nearby_specialists.message ||
    `${data.nearby_specialists.results.length} nearby result(s) found.`;
  if (data.nearby_specialists.maps_url) {
    const mapsLink = document.createElement('a');
    mapsLink.href = data.nearby_specialists.maps_url;
    mapsLink.target = '_blank';
    mapsLink.rel = 'noopener noreferrer';
    mapsLink.textContent = 'Find nearby care on Google Maps';
    nearbyNote.append(document.createElement('br'), mapsLink);
  }
  if (data.nearby_specialists.openstreetmap_url) {
    const osmLink = document.createElement('a');
    osmLink.href = data.nearby_specialists.openstreetmap_url;
    osmLink.target = '_blank';
    osmLink.rel = 'noopener noreferrer';
    osmLink.textContent = 'Open OpenStreetMap';
    nearbyNote.append(document.createElement('br'), osmLink);
  }

  node.querySelector('.explanation-text').textContent = data.advice.explanation;

  const remediesList = node.querySelector('.remedies-list');
  (data.advice.home_remedies || []).forEach((r) => {
    const li = document.createElement('li');
    li.textContent = r;
    remediesList.appendChild(li);
  });

  const redFlagsList = node.querySelector('.redflags-list');
  (data.advice.see_doctor_if || []).forEach((r) => {
    const li = document.createElement('li');
    li.textContent = r;
    redFlagsList.appendChild(li);
  });

  node.querySelector('.ai-disclaimer').textContent = data.advice.disclaimer;

  resultArea.appendChild(node);
}

async function checkSymptoms() {
  const text = input.value.trim();
  if (!text) {
    input.focus();
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Analyzing…';

  try {
    const { latitude, longitude } = await getLocation();
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symptoms: text, latitude, longitude }),
    });
    const data = await res.json();

    if (!res.ok && !data.error) {
      renderError(`The server returned an error (${res.status}). Please try again.`);
    } else if (data.error) {
      renderError(data.error);
    } else {
      renderResult(data);
    }
  } catch (err) {
    renderError('Could not reach the backend. Check that the Flask server is running and open this app through its local/server URL.');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Check symptoms';
  }
}

btn.addEventListener('click', checkSymptoms);
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) checkSymptoms();
});
