/**
 * app.js - Advanced Crash Predictor with Supabase Integration
 * 
 * Key improvements:
 * - Predictions with any number of multipliers (minimum 1)
 * - Enhanced UI/UX with dynamic feedback
 * - Real-time synchronization across devices
 * - Optimized performance for large datasets
 * - Improved error handling and resilience
 * - Secure Supabase integration
 */

// Supabase Configuration
const SUPABASE_URL = "https://fawcuwcqfwzvdoalcocx.supabase.co";
const SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhd2N1d2NxZnd6dmRvYWxjb2N4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTA4NDY3MjYsImV4cCI6MjA2NjQyMjcyNn0.5NCGUTGpPm7w2Jv0GURMKmGh-EQ7WztNLs9MD5_nSjc";

// Initialize Supabase
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY, {
  realtime: { params: { eventsPerSecond: 15 } },
  global: { headers: { 'Content-Type': 'application/json', 'apikey': SUPABASE_KEY } }
});

console.log("Supabase initialized:", supabase);

// Feature detection
const isAdvancedBrowser = () => {
  return window.Worker && window.requestIdleCallback && window.Promise && 
         Array.prototype.includes && Object.values;
};

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
  testSupabaseConnection().then(connected => {
    if (!connected) {
      showPersistentBanner("⚠️ Connection to Supabase failed. Predictions may not save correctly.", "error");
    }
  });
  
  if (!isAdvancedBrowser()) {
    initBasicMode();
    return;
  }
  
  initAdvancedMode();
});

/**
 * Advanced Mode Initialization
 */
function initAdvancedMode() {
  // Performance monitoring
  const performanceMonitor = new PerformanceMonitor();
  performanceMonitor.startMonitoring();
  
  // Prediction engine
  const predictionEngine = new PredictionEngine({
    onPredictionComplete: updateUIWithPrediction,
    onError: handlePredictionError,
    onWorkerReady: () => {
      console.log('✅ Prediction engine initialized');
      document.getElementById('systemStatus').innerHTML = `
        <span class="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
        System Operational (Advanced Mode)
      `;
    }
  });
  
  // UI Manager
  const uiManager = new UIManager(predictionEngine);
  uiManager.initialize();
  window.uiManager = uiManager;
  
  // Voice recognition
  if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const voiceRecognition = new VoiceRecognition(predictionEngine, uiManager);
    voiceRecognition.initialize();
  }
  
  // Real-time multiplier updates
  setupRealtimeMultipliers(uiManager);
  
  // Auto-refresh metadata
  setInterval(() => fetchMetadata(predictionEngine), 300000);
  
  // Initialization tasks
  window.requestIdleCallback(() => {
    predictionEngine.warmup();
    fetchMetadata(predictionEngine);
    uiManager.loadMultipliers();
  });
}

/**
 * Basic Mode Initialization
 */
function initBasicMode() {
  console.log('⚠️ Initializing in basic mode');
  
  document.getElementById('systemStatus').innerHTML = `
    <span class="w-2 h-2 bg-yellow-500 rounded-full mr-2"></span>
    Limited Functionality (Basic Mode)
  `;
  
  // Show browser warning
  const warning = document.createElement('div');
  warning.className = 'bg-yellow-900 border border-yellow-700 text-yellow-100 px-4 py-3 rounded relative mb-4';
  warning.innerHTML = `
    <strong class="font-bold">Browser Warning:</strong>
    <span class="block sm:inline">Your browser doesn't support advanced features. Some functionality may be limited.</span>
  `;
  document.querySelector('.container').insertBefore(warning, document.querySelector('.container').firstChild);
  
  // Initialize basic prediction
  const predictionEngine = new BasicPredictionEngine();
  const uiManager = new UIManager(predictionEngine);
  uiManager.initialize();
  window.uiManager = uiManager;
  
  // Load multipliers
  uiManager.loadMultipliers();
  fetchMetadata(predictionEngine);
}

// Supabase Functions
async function saveMultiplier(value) {
  try {
    const numericValue = parseFloat(value);
    if (isNaN(numericValue)) throw new Error("Invalid number format");
    
    const { data, error } = await supabase
      .from('multipliers')
      .insert([{ 
        value: numericValue,
        created_at: new Date().toISOString()
      }]);
    
    if (error) throw error;
    
    return data[0];
  } catch (error) {
    console.error("Save error:", error);
    throw new Error(`Failed to save multiplier: ${error.message}`);
  }
}

async function getMultipliers(page = 1, pageSize = 500) {
  try {
    const from = (page - 1) * pageSize;
    const to = from + pageSize - 1;
    
    const { data, error, count } = await supabase
      .from('multipliers')
      .select('value, created_at', { count: 'exact' })
      .order('created_at', { ascending: false })
      .range(from, to);
    
    if (error) throw error;
    
    return {
      data,
      totalCount: count,
      currentPage: page,
      totalPages: Math.ceil(count / pageSize)
    };
  } catch (error) {
    console.error("Fetch error:", error);
    throw new Error(`Failed to fetch multipliers: ${error.message}`);
  }
}

async function testSupabaseConnection() {
  try {
    const startTime = performance.now();
    const { error } = await supabase
      .from('multipliers')
      .select('*')
      .limit(1);
    
    const latency = performance.now() - startTime;
    console.log(`Supabase connection test: ${error ? 'Failed' : 'Success'}, Latency: ${latency.toFixed(1)}ms`);
    
    return !error;
  } catch (error) {
    console.error("Connection test failed:", error);
    return false;
  }
}

function setupRealtimeMultipliers(uiManager) {
  const subscription = supabase
    .channel('multiplier-changes')
    .on('postgres_changes', {
      event: 'INSERT',
      schema: 'public',
      table: 'multipliers'
    }, (payload) => {
      console.log('New multiplier added:', payload.new);
      uiManager.addMultiplierLocally(payload.new.value);
    })
    .subscribe();

  window.addEventListener('beforeunload', () => {
    supabase.removeChannel(subscription);
  });
}

/**
 * Prediction Engine with Web Workers
 */
class PredictionEngine {
  constructor(options = {}) {
    this.options = options;
    this.worker = null;
    this.isWorkerReady = false;
    this.accuracy = 75.0;
    this.multiplierRange = 20.0;
    this.lastPrediction = null;
    this.performanceStats = { avgCalculationTime: 0, calculationCount: 0 };
    
    this.initializeWorker();
    window.addEventListener('beforeunload', () => this.cleanup());
  }
  
  initializeWorker() {
    const workerCode = `
      self.addEventListener('message', (e) => {
        const { type, data } = e.data;
        
        if (type === 'calculatePrediction') {
          const startTime = performance.now();
          const result = calculatePrediction(data.multipliers, data.accuracy, data.range);
          const calculationTime = performance.now() - startTime;
          
          self.postMessage({ type: 'predictionResult', data: result, calculationTime });
        } 
        else if (type === 'warmup') {
          calculatePrediction(Array(50).fill(1.5), 75.0, 20.0);
          self.postMessage({ type: 'warmupComplete' });
        }
      });
      
      function calculatePrediction(multipliers, accuracy, multiplierRange) {
        if (!multipliers || multipliers.length === 0) {
          return { error: 'Need at least 1 multiplier for prediction' };
        }
        
        // Use last 50 multipliers or all available if less
        const sampleSize = Math.min(multipliers.length, 50);
        const recentMultipliers = multipliers.slice(-sampleSize);
        const avg = recentMultipliers.reduce((sum, val) => sum + val, 0) / sampleSize;
        
        // Apply accuracy-based weighting
        const confidence = accuracy / 100;
        const randomFactor = 0.95 + (Math.random() * 0.1 * (1 - confidence));
        let predictedValue = avg * randomFactor;
        
        // Ensure predicted value is reasonable
        predictedValue = Math.max(1.0, Math.min(10.0, predictedValue));
        
        // Calculate range
        const rangeLower = Math.max(1.0, predictedValue * (1 - multiplierRange / 100));
        const rangeUpper = predictedValue * (1 + multiplierRange / 100);
        
        // Determine risk level
        let riskLevel = 'Low';
        if (multiplierRange > 30) riskLevel = 'High';
        else if (multiplierRange > 15) riskLevel = 'Medium';
        
        return { predictedValue, confidence, rangeLower, rangeUpper, riskLevel };
      }
    `;
    
    try {
      const blob = new Blob([workerCode], {type: 'application/javascript'});
      this.worker = new Worker(URL.createObjectURL(blob));
      
      this.worker.onmessage = (e) => {
        const { type, data, calculationTime } = e.data;
        
        if (type === 'predictionResult') {
          this.handlePredictionResult(data, calculationTime);
        } 
        else if (type === 'warmupComplete') {
          this.isWorkerReady = true;
          this.options.onWorkerReady?.();
        }
      };
      
      this.worker.onerror = (error) => {
        console.error('Worker error:', error);
        this.fallbackToBasicEngine();
      };
    } catch (e) {
      console.error('Worker initialization failed:', e);
      this.fallbackToBasicEngine();
    }
  }
  
  calculatePrediction(multipliers) {
    if (!this.isWorkerReady || !this.worker) {
      this.fallbackCalculatePrediction(multipliers);
      return;
    }
    
    this.worker.postMessage({
      type: 'calculatePrediction',
      data: { multipliers, accuracy: this.accuracy, range: this.multiplierRange }
    });
  }
  
  handlePredictionResult(result, calculationTime) {
    if (result.error) {
      this.options.onError?.(result.error);
      return;
    }
    
    // Update performance stats
    this.performanceStats.calculationCount++;
    const total = this.performanceStats.avgCalculationTime * (this.performanceStats.calculationCount - 1) + calculationTime;
    this.performanceStats.avgCalculationTime = total / this.performanceStats.calculationCount;
    
    // Store last prediction
    this.lastPrediction = { ...result, timestamp: new Date().toISOString() };
    
    // Notify callback
    this.options.onPredictionComplete?.(this.lastPrediction);
  }
  
  warmup() {
    if (this.worker && this.isWorkerReady) {
      this.worker.postMessage({ type: 'warmup' });
    }
  }
  
  updateAccuracy(accuracy, multiplierRange) {
    this.accuracy = accuracy;
    this.multiplierRange = multiplierRange;
  }
  
  fallbackToBasicEngine() {
    console.log('⚠️ Falling back to basic prediction engine');
    this.cleanup();
    this.isWorkerReady = false;
    this.calculatePrediction = this.fallbackCalculatePrediction;
  }
  
  cleanup() {
    this.worker?.terminate();
    this.worker = null;
  }
  
  fallbackCalculatePrediction(multipliers) {
    const startTime = performance.now();
    
    try {
      if (!multipliers || multipliers.length === 0) {
        throw new Error('Need at least 1 multiplier for prediction');
      }
      
      // Use last 50 multipliers or all available if less
      const sampleSize = Math.min(multipliers.length, 50);
      const recentMultipliers = multipliers.slice(-sampleSize);
      const avg = recentMultipliers.reduce((sum, val) => sum + val, 0) / sampleSize;
      
      // Apply accuracy-based weighting
      const confidence = this.accuracy / 100;
      const randomFactor = 0.95 + (Math.random() * 0.1 * (1 - confidence));
      let predictedValue = avg * randomFactor;
      
      // Ensure predicted value is reasonable
      predictedValue = Math.max(1.0, Math.min(10.0, predictedValue));
      
      // Calculate range
      const rangeLower = Math.max(1.0, predictedValue * (1 - this.multiplierRange / 100));
      const rangeUpper = predictedValue * (1 + this.multiplierRange / 100);
      
      // Determine risk level
      let riskLevel = 'Low';
      if (this.multiplierRange > 30) riskLevel = 'High';
      else if (this.multiplierRange > 15) riskLevel = 'Medium';
      
      const result = { predictedValue, confidence, rangeLower, rangeUpper, riskLevel };
      this.handlePredictionResult(result, performance.now() - startTime);
    } catch (error) {
      this.options.onError?.(error.message);
    }
  }
}

/**
 * Basic Prediction Engine (fallback)
 */
class BasicPredictionEngine {
  constructor() {
    this.accuracy = 75.0;
    this.multiplierRange = 20.0;
    this.lastPrediction = null;
  }
  
  calculatePrediction(multipliers) {
    const startTime = performance.now();
    
    try {
      if (!multipliers || multipliers.length === 0) {
        throw new Error('Need at least 1 multiplier for prediction');
      }
      
      // Use last 50 multipliers or all available if less
      const sampleSize = Math.min(multipliers.length, 50);
      const recentMultipliers = multipliers.slice(-sampleSize);
      const avg = recentMultipliers.reduce((sum, val) => sum + val, 0) / sampleSize;
      
      // Apply accuracy-based weighting
      const confidence = this.accuracy / 100;
      const randomFactor = 0.95 + (Math.random() * 0.1 * (1 - confidence));
      let predictedValue = avg * randomFactor;
      
      // Ensure predicted value is reasonable
      predictedValue = Math.max(1.0, Math.min(10.0, predictedValue));
      
      // Calculate range
      const rangeLower = Math.max(1.0, predictedValue * (1 - this.multiplierRange / 100));
      const rangeUpper = predictedValue * (1 + this.multiplierRange / 100);
      
      // Determine risk level
      let riskLevel = 'Low';
      if (this.multiplierRange > 30) riskLevel = 'High';
      else if (this.multiplierRange > 15) riskLevel = 'Medium';
      
      const result = {
        predictedValue,
        confidence,
        rangeLower,
        rangeUpper,
        riskLevel,
        calculationTime: performance.now() - startTime
      };
      
      window.updateUIWithPrediction?.(result);
    } catch (error) {
      window.handlePredictionError?.(error.message);
    }
  }
  
  updateAccuracy(accuracy, multiplierRange) {
    this.accuracy = accuracy;
    this.multiplierRange = multiplierRange;
  }
}

/**
 * UI Manager
 */
class UIManager {
  constructor(predictionEngine) {
    this.predictionEngine = predictionEngine;
    this.multipliers = [];
    this.accuracy = 75.0;
    this.multiplierRange = 20.0;
    this.isLoading = false;
    this.currentPage = 1;
    this.totalPages = 1;
    
    // Cache DOM elements
    this.elements = {
      multiplierInput: document.getElementById('multiplierInput'),
      addMultiplierBtn: document.getElementById('addMultiplier'),
      generatePredictionBtn: document.getElementById('generatePrediction'),
      multipliersContainer: document.getElementById('multipliersContainer'),
      multiplierCount: document.getElementById('multiplierCount'),
      noMultipliersMessage: document.getElementById('noMultipliersMessage'),
      predictionResults: document.getElementById('predictionResults'),
      predictedValue: document.getElementById('predictedValue'),
      rangeLowerValue: document.getElementById('rangeLowerValue'),
      rangeUpperValue: document.getElementById('rangeUpperValue'),
      confidenceBar: document.getElementById('confidenceBar'),
      confidencePercentage: document.getElementById('confidencePercentage'),
      riskLevel: document.getElementById('riskLevel'),
      loadingIndicator: document.getElementById('loadingIndicator'),
      loadMoreBtn: document.getElementById('loadMoreMultipliers')
    };
  }
  
  initialize() {
    this.setupEventListeners();
    this.updateAccuracyDisplay();
    this.updateRangeDisplay();
    this.scrollToBottom();
    this.updatePredictionButtonState();
  }
  
  setupEventListeners() {
    this.elements.addMultiplierBtn.addEventListener('click', () => this.addMultiplier());
    this.elements.multiplierInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') this.addMultiplier();
    });
    this.elements.generatePredictionBtn.addEventListener('click', () => this.generatePrediction());
    
    if (this.elements.loadMoreBtn) {
      this.elements.loadMoreBtn.addEventListener('click', () => this.loadMoreMultipliers());
    }
  }
  
  async loadMultipliers() {
    if (this.isLoading) return;
    
    this.showLoading(true, "Loading multipliers...");
    
    try {
      const { data, totalCount, currentPage, totalPages } = await getMultipliers(this.currentPage);
      this.totalPages = totalPages;
      
      // Add new multipliers to the list
      this.multipliers = [...data.map(m => m.value), ...this.multipliers];
      this.updateMultipliersDisplay();
      
      // Update multiplier count
      this.elements.multiplierCount.textContent = totalCount;
      
      // Update load more button
      if (this.elements.loadMoreBtn) {
        this.elements.loadMoreBtn.style.display = 
          this.currentPage < this.totalPages ? 'block' : 'none';
      }
      
      this.updatePredictionButtonState();
      
    } catch (error) {
      console.error('Error loading multipliers:', error);
      showSpeechBubble('Failed to load multipliers. Please try again.');
    } finally {
      this.showLoading(false);
    }
  }
  
  async loadMoreMultipliers() {
    if (this.isLoading || this.currentPage >= this.totalPages) return;
    
    this.currentPage++;
    this.showLoading(true, "Loading more multipliers...");
    
    try {
      const { data } = await getMultipliers(this.currentPage);
      this.multipliers = [...data.map(m => m.value), ...this.multipliers];
      this.updateMultipliersDisplay();
      
      // Update load more button
      if (this.elements.loadMoreBtn) {
        this.elements.loadMoreBtn.style.display = 
          this.currentPage < this.totalPages ? 'block' : 'none';
      }
      
    } catch (error) {
      console.error('Error loading more multipliers:', error);
      this.currentPage--;
      showSpeechBubble('Failed to load more multipliers. Please try again.');
    } finally {
      this.showLoading(false);
    }
  }
  
  addMultiplierLocally(value) {
    this.multipliers.push(value);
    this.updateMultipliersDisplay();
    this.updatePredictionButtonState();
    this.scrollToBottom();
  }
  
  async addMultiplier() {
    const value = this.elements.multiplierInput.value.trim();
    if (value === '') return;
    
    const numericValue = parseFloat(value);
    if (isNaN(numericValue) || numericValue <= 0) {
      showSpeechBubble('Please enter a valid positive number');
      return;
    }
    
    this.showLoading(true, "Saving multiplier...");
    
    try {
      await saveMultiplier(numericValue);
      this.addMultiplierLocally(numericValue);
      this.elements.multiplierInput.value = '';
      this.elements.multiplierInput.focus();
    } catch (error) {
      console.error('Error saving multiplier:', error);
      showSpeechBubble('Failed to save multiplier. Please try again.');
    } finally {
      this.showLoading(false);
    }
  }
  
  deleteMultiplier(index) {
    if (index >= 0 && index < this.multipliers.length) {
      this.multipliers.splice(index, 1);
      this.updateMultipliersDisplay();
      this.updatePredictionButtonState();
    }
  }
  
  updateMultipliersDisplay() {
    // Update count
    this.elements.multiplierCount.textContent = this.multipliers.length;
    
    // Show/hide no multipliers message
    this.elements.noMultipliersMessage.style.display = 
      this.multipliers.length === 0 ? 'block' : 'none';
    
    // Clear container
    this.elements.multipliersContainer.innerHTML = '';
    
    // Add multipliers (newest first)
    for (let i = this.multipliers.length - 1; i >= 0; i--) {
      const multiplier = this.multipliers[i];
      
      const multiplierItem = document.createElement('div');
      multiplierItem.className = 'multiplier-item bg-gray-800 rounded p-2 mb-2 flex items-center';
      multiplierItem.innerHTML = `
        <span class="flex-1 font-mono">${multiplier.toFixed(2)}</span>
        <button class="delete-btn text-red-400 hover:text-red-300 transition duration-200 ml-2">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
      `;
      
      // Add delete functionality
      const deleteBtn = multiplierItem.querySelector('.delete-btn');
      deleteBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.deleteMultiplier(i);
      });
      
      this.elements.multipliersContainer.appendChild(multiplierItem);
    }
  }
  
  generatePrediction() {
    if (this.multipliers.length === 0) {
      showSpeechBubble('Add at least 1 multiplier to generate prediction');
      return;
    }
    
    // Show loading state
    this.elements.generatePredictionBtn.disabled = true;
    this.elements.generatePredictionBtn.innerHTML = `
      <svg class="animate-spin h-4 w-4 mr-2 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
      </svg> Predicting...
    `;
    
    // Calculate prediction
    this.predictionEngine.calculatePrediction(this.multipliers);
  }
  
  updateUIWithPrediction(prediction) {
    // Update prediction values
    this.elements.predictedValue.textContent = prediction.predictedValue.toFixed(2);
    this.elements.rangeLowerValue.textContent = prediction.rangeLower.toFixed(2);
    this.elements.rangeUpperValue.textContent = prediction.rangeUpper.toFixed(2);
    
    // Update confidence
    const confidencePercent = Math.round(prediction.confidence * 100);
    this.elements.confidencePercentage.textContent = `${confidencePercent}%`;
    this.elements.confidenceBar.style.width = `${confidencePercent}%`;
    
    // Update risk level
    this.elements.riskLevel.textContent = prediction.riskLevel;
    
    // Visual feedback
    const riskColor = prediction.riskLevel === 'High' ? 'text-red-500' : 
                     prediction.riskLevel === 'Medium' ? 'text-yellow-500' : 'text-green-500';
    this.elements.riskLevel.className = `text-sm font-semibold ${riskColor}`;
    
    // Re-enable button
    this.elements.generatePredictionBtn.disabled = false;
    this.elements.generatePredictionBtn.textContent = 'Generate Prediction';
    
    // Show prediction results
    this.elements.predictionResults.classList.add('visible');
    
    // Show speech bubble
    showSpeechBubble(
      `Prediction: ${prediction.predictedValue.toFixed(2)} | Range: ${prediction.rangeLower.toFixed(2)}-${prediction.rangeUpper.toFixed(2)}`,
      5000
    );
    
    // Add warning if few multipliers
    if (this.multipliers.length < 50) {
      showSpeechBubble(
        `Warning: Prediction accuracy may be low with only ${this.multipliers.length} multipliers (recommended: 50+)`,
        5000
      );
    }
  }
  
  updateAccuracyDisplay() {
    this.elements.accuracyDisplay?.textContent = `${this.accuracy.toFixed(2)}%`;
    document.getElementById('accuracyValue')?.textContent = `${Math.round(this.accuracy)}%`;
    document.getElementById('confidenceMeter')?.style.width = `${this.accuracy}%`;
  }
  
  updateRangeDisplay() {
    this.elements.rangeDisplay?.textContent = `┬▒${this.multiplierRange.toFixed(2)}%`;
    const volatilityPercent = Math.min(100, this.multiplierRange * 1.5);
    document.getElementById('volatilityMeter')?.style.width = `${volatilityPercent}%`;
  }
  
  updatePredictionButtonState() {
    this.elements.generatePredictionBtn.disabled = this.multipliers.length === 0;
    this.elements.generatePredictionBtn.textContent = 
      this.multipliers.length > 0 ? 'Generate Prediction' : 'Add Multipliers First';
  }
  
  showLoading(isLoading, message = "") {
    this.isLoading = isLoading;
    if (this.elements.loadingIndicator) {
      this.elements.loadingIndicator.style.display = isLoading ? 'block' : 'none';
      if (message && this.elements.loadingIndicator.querySelector('.loading-text')) {
        this.elements.loadingIndicator.querySelector('.loading-text').textContent = message;
      }
    }
  }
  
  scrollToBottom() {
    this.elements.multipliersContainer.scrollTop = this.elements.multipliersContainer.scrollHeight;
  }
}

/**
 * Voice Recognition
 */
class VoiceRecognition {
  constructor(predictionEngine, uiManager) {
    this.predictionEngine = predictionEngine;
    this.uiManager = uiManager;
    this.speechRecognition = null;
    this.isListening = false;
    this.voiceToggle = document.getElementById('voiceToggle');
    this.voiceIndicator = document.getElementById('voiceIndicator');
  }
  
  initialize() {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      this.voiceToggle.disabled = true;
      this.voiceToggle.parentElement.parentElement.innerHTML = 'Voice recognition is not supported in your browser.';
      return;
    }
    
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    this.speechRecognition = new SpeechRecognition();
    this.speechRecognition.continuous = true;
    this.speechRecognition.interimResults = true;
    this.speechRecognition.lang = 'en-US';
    
    this.speechRecognition.onstart = () => {
      this.isListening = true;
      this.voiceIndicator.classList.add('active');
      this.voiceToggle.checked = true;
      localStorage.setItem('voiceRecognitionEnabled', 'true');
      showSpeechBubble('Listening for commands...');
    };
    
    this.speechRecognition.onend = () => {
      this.isListening = false;
      this.voiceIndicator.classList.remove('active');
      if (!this.voiceToggle.checked) {
        localStorage.setItem('voiceRecognitionEnabled', 'false');
      }
      if (this.voiceToggle.checked) {
        try { this.speechRecognition.start(); } catch (e) { console.error('Restart error:', e); }
      }
    };
    
    this.speechRecognition.onresult = (event) => {
      let transcript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) transcript += event.results[i][0].transcript;
      }
      if (transcript.trim() !== '') this.processVoiceCommand(transcript.toLowerCase());
    };
    
    this.speechRecognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      if (event.error === 'not-allowed') {
        showSpeechBubble('Microphone access denied. Please enable permissions.');
        this.voiceToggle.checked = false;
        localStorage.setItem('voiceRecognitionEnabled', 'false');
      }
    };
    
    this.voiceToggle.addEventListener('change', () => {
      localStorage.setItem('voiceRecognitionEnabled', this.voiceToggle.checked);
      this.toggleVoiceRecognition();
    });
    
    if (localStorage.getItem('voiceRecognitionEnabled') === 'true') {
      this.voiceToggle.checked = true;
      try {
        this.speechRecognition.start();
        this.isListening = true;
        this.voiceIndicator.classList.add('active');
      } catch (e) {
        console.error('Start error:', e);
      }
    }
  }
  
  toggleVoiceRecognition() {
    if (!this.speechRecognition) return;
    
    if (this.voiceToggle.checked) {
      try {
        this.speechRecognition.start();
        showSpeechBubble('Voice recognition activated');
      } catch (e) {
        console.error('Start error:', e);
        this.voiceToggle.checked = false;
        localStorage.setItem('voiceRecognitionEnabled', 'false');
        showSpeechBubble('Error starting voice recognition. Please try again.');
      }
    } else {
      try {
        this.speechRecognition.stop();
        showSpeechBubble('Voice recognition deactivated');
      } catch (e) {
        console.error('Stop error:', e);
      }
    }
  }
  
  processVoiceCommand(command) {
    console.log('Voice command:', command);
    
    if (command.includes('multiplier') && command.match(/[\d.]+/)) {
      const value = parseFloat(command.match(/[\d.]+/)[0]);
      if (!isNaN(value) && value > 0) this.uiManager.addMultiplier(value);
    } 
    else if (command.includes('delete last') && this.uiManager.multipliers.length > 0) {
      const lastMultiplier = this.uiManager.multipliers.pop();
      this.uiManager.updateMultipliersDisplay();
      showSpeechBubble(`Deleted multiplier ${lastMultiplier}`);
    } 
    else if (command.includes('predict') || command.includes('calculate')) {
      this.uiManager.generatePrediction();
    }
  }
}

/**
 * Performance Monitor
 */
class PerformanceMonitor {
  constructor() {
    this.fps = 0;
    this.lastFpsUpdate = 0;
    this.fpsSamples = [];
    this.frameCount = 0;
    this.lastFrameTime = 0;
  }
  
  startMonitoring() {
    this.lastFpsUpdate = performance.now();
    this.lastFrameTime = this.lastFpsUpdate;
    
    const monitorFrame = () => {
      const now = performance.now();
      this.frameCount++;
      
      if (now > this.lastFpsUpdate + 1000) {
        this.fps = Math.round((this.frameCount * 1000) / (now - this.lastFpsUpdate));
        this.fpsSamples.push(this.fps);
        if (this.fpsSamples.length > 10) this.fpsSamples.shift();
        this.frameCount = 0;
        this.lastFpsUpdate = now;
      }
      
      this.lastFrameTime = now;
      requestAnimationFrame(monitorFrame);
    };
    
    requestAnimationFrame(monitorFrame);
  }
}

// Helper Functions
function showSpeechBubble(message, duration = 3000) {
  const speechBubble = document.getElementById('speechBubble');
  if (!speechBubble) return;
  
  speechBubble.textContent = message;
  speechBubble.classList.add('visible');
  
  setTimeout(() => speechBubble.classList.remove('visible'), duration);
}

function showPersistentBanner(message, type = "info") {
  const banner = document.createElement('div');
  banner.className = `banner ${type}-banner fixed top-4 left-1/2 transform -translate-x-1/2 p-4 rounded-md z-50 shadow-lg`;
  banner.innerHTML = `
    <div class="flex items-center">
      <span class="mr-2">${message}</span>
      <button class="close-banner ml-4 text-lg">&times;</button>
    </div>
  `;
  
  document.body.appendChild(banner);
  banner.querySelector('.close-banner').addEventListener('click', () => banner.remove());
  setTimeout(() => banner.remove(), 10000);
}

function fetchMetadata(predictionEngine) {
  fetch('https://raw.githubusercontent.com/eustancek/Crashpredictor/main/metadata.json')
    .then(response => {
      if (!response.ok) throw new Error('Network error');
      return response.json();
    })
    .then(data => {
      const accuracy = data.accuracy || 75.0;
      const multiplierRange = data.range_percentage || 20.0;
      predictionEngine.updateAccuracy(accuracy, multiplierRange);
      
      if (window.uiManager) {
        window.uiManager.accuracy = accuracy;
        window.uiManager.multiplierRange = multiplierRange;
        window.uiManager.updateAccuracyDisplay();
        window.uiManager.updateRangeDisplay();
      }
    })
    .catch(console.error);
}

// Global functions
window.updateUIWithPrediction = function(prediction) {
  window.uiManager?.updateUIWithPrediction(prediction);
};

window.handlePredictionError = function(error) {
  showSpeechBubble(`Prediction error: ${error}`);
  if (window.uiManager?.elements?.generatePredictionBtn) {
    window.uiManager.elements.generatePredictionBtn.disabled = false;
    window.uiManager.elements.generatePredictionBtn.textContent = 'Generate Prediction';
  }
};
