// Main Application Logic
import { CrashPredictor } from './model.js';
import { initVoice } from './voice.js';
import { updateAccuracyChart } from './charts.js';

const supabase = createClient(
  'YOUR_SUPABASE_URL',
  'YOUR_SUPABASE_ANON_KEY'
);

let predictor = new CrashPredictor();
let currentMode = 'prediction';

document.addEventListener('DOMContentLoaded', async () => {
    await predictor.init();
    initVoice();
    setupEventListeners();
    syncWithSupabase();
});

function setupEventListeners() {
    // Mode Switching
    document.getElementById('training-mode').addEventListener('click', () => {
        currentMode = 'training';
        document.getElementById('mode-status').textContent = 'Training Mode';
    });

    document.getElementById('prediction-mode').addEventListener('click', () => {
        currentMode = 'prediction';
        document.getElementById('mode-status').textContent = 'Prediction Mode';
    });

    // Analyze Button
    document.getElementById('analyze').addEventListener('click', async () => {
        const multipliers = getValuesFromTextarea('multipliers');
        const crashValues = getValuesFromTextarea('crash-values');
        
        const result = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({multipliers, crashValues})
        }).then(r => r.json());
        
        displayPrediction(result);
        updateAccuracyChart(result.accuracy);
        
        if (currentMode === 'training') {
            saveTrainingHistory(multipliers, crashValues, result);
        }
    });

    // Data Deletion
    document.querySelectorAll('[data-delete]').forEach(btn => {
        btn.addEventListener('click', () => {
            const type = btn.dataset.delete;
            if (confirm(`Delete all ${type}? This won't affect learning`)) {
                deleteAllData(type);
            }
        });
    });
}

async function syncWithSupabase() {
    // Delta sync implementation
    const lastSync = localStorage.getItem('last_sync') || '2023-01-01';
    
    // Upload new data
    const newMultipliers = await getLocalDataSince(lastSync, 'multipliers');
    const newCrashValues = await getLocalDataSince(lastSync, 'crash-values');
    
    await supabase.from('multipliers').upsert(newMultipliers);
    await supabase.from('crash-values').upsert(newCrashValues);
    
    localStorage.setItem('last_sync', new Date().toISOString());
}