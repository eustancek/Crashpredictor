/**
 * app.js - Advanced Prediction Engine for Crash Predictor
 * 
 * Key improvements:
 * - Fixed worker scope bug in calculation function
 * - Added proper worker termination to prevent memory leaks
 * - Enhanced error handling for worker initialization
 * - Added page unload cleanup
 * - Improved UI update consistency
 * - Optimized fallback logic
 * 
 * Designed specifically for Netlify deployment with zero freezing
 */

// Feature detection for critical APIs
const isAdvancedBrowser = () => {
  return window.Worker && window.requestIdleCallback && window.Promise && 
         Array.prototype.includes && Object.values;
};

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
  if (!isAdvancedBrowser()) {
    initBasicMode();
    return;
  }
  
  initAdvancedMode();
});

/**
 * Advanced mode - uses Web Workers and advanced APIs
 */
function initAdvancedMode() {
  // Performance monitoring
  const performanceMonitor = new PerformanceMonitor();
  performanceMonitor.startMonitoring();
  
  // Create prediction engine with Web Worker
  const predictionEngine = new PredictionEngine({
    onPredictionComplete: updateUIWithPrediction,
    onError: handlePredictionError,
    onWorkerReady: () => {
      console.log('✅ Prediction engine initialized with Web Worker');
      document.getElementById('systemStatus').innerHTML = `
        <span class="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
        System Operational (Advanced Mode)
      `;
    }
  });
  
  // Initialize UI components
  const uiManager = new UIManager(predictionEngine);
  uiManager.initialize();
  window.uiManager = uiManager; // Make accessible globally
  
  // Setup voice recognition if available
  if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const voiceRecognition = new VoiceRecognition(predictionEngine, uiManager);
    voiceRecognition.initialize();
  }
  
  // Auto-refresh metadata periodically
  setInterval(() => {
    fetchMetadata(predictionEngine);
  }, 60000); // Every minute
  
  // Performance optimization: Use requestIdleCallback for non-urgent tasks
  window.requestIdleCallback(() => {
    // Pre-warm the prediction engine with a dummy calculation
    predictionEngine.warmup();
    
    // Load metadata immediately
    fetchMetadata(predictionEngine);
  });
}

/**
 * Basic mode - fallback for older browsers
 */
function initBasicMode() {
  console.log('⚠️ Initializing in basic mode (older browser)');
  
  document.getElementById('systemStatus').innerHTML = `
    <span class="w-2 h-2 bg-yellow-500 rounded-full mr-2"></span>
    Limited Functionality (Basic Mode)
  `;
  
  // Show warning to user
  const warning = document.createElement('div');
  warning.className = 'bg-yellow-900 border border-yellow-700 text-yellow-100 px-4 py-3 rounded relative mb-4';
  warning.innerHTML = `
    <strong class="font-bold">Browser Warning:</strong>
    <span class="block sm:inline">Your browser doesn't support advanced features. Some functionality may be limited.</span>
  `;
  document.querySelector('.container').insertBefore(warning, document.querySelector('.container').firstChild);
  
  // Initialize with simplified prediction logic
  const predictionEngine = new BasicPredictionEngine();
  const uiManager = new UIManager(predictionEngine);
  uiManager.initialize();
  window.uiManager = uiManager; // Make accessible globally
  
  // Load metadata
  fetchMetadata(predictionEngine);
}

/**
 * Web Worker-based Prediction Engine
 */
class PredictionEngine {
  constructor(options = {}) {
    this.options = options;
    this.worker = null;
    this.isWorkerReady = false;
    this.accuracy = 75.0;
    this.multiplierRange = 20.0;
    this.lastPrediction = null;
    this.performanceStats = {
      avgCalculationTime: 0,
      calculationCount: 0
    };
    
    this.initializeWorker();
    
    // Clean up worker when page unloads
    window.addEventListener('beforeunload', () => {
      if (this.worker) {
        this.worker.terminate();
      }
    });
  }
  
  initializeWorker() {
    // Create Web Worker from inline code
    const workerCode = `
      // Prediction worker code
      function calculatePrediction(multipliers, accuracy, multiplierRange) {
        // Validate input
        if (!multipliers || multipliers.length < 50) {
          return { error: 'Need at least 50 multipliers for prediction' };
        }
        
        // Calculate prediction
        const recentMultipliers = multipliers.slice(-50);
        const avg = recentMultipliers.reduce((sum, val) => sum + val, 0) / 50;
        
        // Apply accuracy-based weighting
        const confidence = accuracy / 100;
        const randomFactor = 0.95 + (Math.random() * 0.1 * (1 - confidence));
        let predictedValue = avg * randomFactor;
        
        // Ensure predicted value is reasonable
        predictedValue = Math.max(1.0, Math.min(10.0, predictedValue));
        
        // Calculate range based on multiplierRange
        const rangeLower = Math.max(1.0, predictedValue * (1 - multiplierRange / 100));
        const rangeUpper = predictedValue * (1 + multiplierRange / 100);
        
        // Determine risk level
        let riskLevel = 'Low';
        if (multiplierRange > 30) {
          riskLevel = 'High';
        } else if (multiplierRange > 15) {
          riskLevel = 'Medium';
        }
        
        return {
          predictedValue: predictedValue,
          confidence: confidence,
          rangeLower: rangeLower,
          rangeUpper: rangeUpper,
          riskLevel: riskLevel
        };
      }

      self.addEventListener('message', function(e) {
        const { type, data } = e.data;
        
        if (type === 'calculatePrediction') {
          const startTime = performance.now();
          const result = calculatePrediction(data.multipliers, data.accuracy, data.range);
          const calculationTime = performance.now() - startTime;
          
          self.postMessage({
            type: 'predictionResult',
            data: result,
            calculationTime: calculationTime
          });
        } else if (type === 'warmup') {
          // Pre-warm the prediction engine
          const dummyMultipliers = Array(50).fill(1.5);
          calculatePrediction(dummyMultipliers, 75.0, 20.0);
          self.postMessage({ type: 'warmupComplete' });
        }
      });
    `;
    
    try {
      // Create blob URL for the worker
      const blob = new Blob([workerCode], {type: 'application/javascript'});
      this.worker = new Worker(URL.createObjectURL(blob));
      
      // Handle worker messages
      this.worker.onmessage = (e) => {
        const { type, data, calculationTime } = e.data;
        
        if (type === 'predictionResult') {
          this.handlePredictionResult(data, calculationTime);
        } else if (type === 'warmupComplete') {
          this.isWorkerReady = true;
          if (this.options.onWorkerReady) {
            this.options.onWorkerReady();
          }
        }
      };
      
      // Handle worker errors
      this.worker.onerror = (error) => {
        console.error('Prediction worker error:', error);
        this.fallbackToBasicEngine();
      };
    } catch (e) {
      console.error('Worker initialization failed:', e);
      this.fallbackToBasicEngine();
    }
  }
  
  calculatePrediction(multipliers) {
    if (!this.isWorkerReady || !this.worker) {
      console.warn('Worker not ready, using fallback');
      this.fallbackCalculatePrediction(multipliers);
      return;
    }
    
    // Send calculation to worker
    this.worker.postMessage({
      type: 'calculatePrediction',
      data: {
        multipliers: multipliers,
        accuracy: this.accuracy,
        range: this.multiplierRange
      }
    });
  }
  
  handlePredictionResult(result, calculationTime) {
    if (result.error) {
      if (this.options.onError) {
        this.options.onError(result.error);
      }
      return;
    }
    
    // Update performance stats
    this.updatePerformanceStats(calculationTime);
    
    // Store last prediction
    this.lastPrediction = {
      ...result,
      timestamp: new Date().toISOString()
    };
    
    // Notify callback
    if (this.options.onPredictionComplete) {
      this.options.onPredictionComplete(this.lastPrediction);
    }
  }
  
  updatePerformanceStats(calculationTime) {
    this.performanceStats.calculationCount++;
    const total = this.performanceStats.avgCalculationTime * (this.performanceStats.calculationCount - 1) + calculationTime;
    this.performanceStats.avgCalculationTime = total / this.performanceStats.calculationCount;
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
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.isWorkerReady = false;
    
    // Replace methods with basic implementations
    this.calculatePrediction = this.fallbackCalculatePrediction;
  }
  
  fallbackCalculatePrediction(multipliers) {
    // Basic prediction without Web Worker
    const startTime = performance.now();
    
    try {
      // Calculate prediction
      const recentMultipliers = multipliers.slice(-50);
      const avg = recentMultipliers.reduce((sum, val) => sum + val, 0) / 50;
      
      // Apply accuracy-based weighting
      const confidence = this.accuracy / 100;
      const randomFactor = 0.95 + (Math.random() * 0.1 * (1 - confidence));
      let predictedValue = avg * randomFactor;
      
      // Ensure predicted value is reasonable
      predictedValue = Math.max(1.0, Math.min(10.0, predictedValue));
      
      // Calculate range based on multiplierRange
      const rangeLower = Math.max(1.0, predictedValue * (1 - this.multiplierRange / 100));
      const rangeUpper = predictedValue * (1 + this.multiplierRange / 100);
      
      // Determine risk level
      let riskLevel = 'Low';
      if (this.multiplierRange > 30) {
        riskLevel = 'High';
      } else if (this.multiplierRange > 15) {
        riskLevel = 'Medium';
      }
      
      const result = {
        predictedValue: predictedValue,
        confidence: confidence,
        rangeLower: rangeLower,
        rangeUpper: rangeUpper,
        riskLevel: riskLevel
      };
      
      const calculationTime = performance.now() - startTime;
      this.handlePredictionResult(result, calculationTime);
    } catch (error) {
      if (this.options.onError) {
        this.options.onError(error.message);
      }
    }
  }
}

/**
 * Basic Prediction Engine (for older browsers)
 */
class BasicPredictionEngine {
  constructor() {
    this.accuracy = 75.0;
    this.multiplierRange = 20.0;
    this.lastPrediction = null;
  }
  
  calculatePrediction(multipliers) {
    // Basic prediction without Web Worker
    const startTime = performance.now();
    
    try {
      // Calculate prediction
      const recentMultipliers = multipliers.slice(-50);
      const avg = recentMultipliers.reduce((sum, val) => sum + val, 0) / 50;
      
      // Apply accuracy-based weighting
      const confidence = this.accuracy / 100;
      const randomFactor = 0.95 + (Math.random() * 0.1 * (1 - confidence));
      let predictedValue = avg * randomFactor;
      
      // Ensure predicted value is reasonable
      predictedValue = Math.max(1.0, Math.min(10.0, predictedValue));
      
      // Calculate range based on multiplierRange
      const rangeLower = Math.max(1.0, predictedValue * (1 - this.multiplierRange / 100));
      const rangeUpper = predictedValue * (1 + this.multiplierRange / 100);
      
      // Determine risk level
      let riskLevel = 'Low';
      if (this.multiplierRange > 30) {
        riskLevel = 'High';
      } else if (this.multiplierRange > 15) {
        riskLevel = 'Medium';
      }
      
      const result = {
        predictedValue: predictedValue,
        confidence: confidence,
        rangeLower: rangeLower,
        rangeUpper: rangeUpper,
        riskLevel: riskLevel,
        calculationTime: performance.now() - startTime
      };
      
      if (window.updateUIWithPrediction) {
        window.updateUIWithPrediction(result);
      }
    } catch (error) {
      console.error('Prediction error:', error);
      if (window.handlePredictionError) {
        window.handlePredictionError(error.message);
      }
    }
  }
  
  updateAccuracy(accuracy, multiplierRange) {
    this.accuracy = accuracy;
    this.multiplierRange = multiplierRange;
  }
}

/**
 * UI Manager - handles all UI interactions
 */
class UIManager {
  constructor(predictionEngine) {
    this.predictionEngine = predictionEngine;
    this.multipliers = [];
    this.accuracy = 75.0;
    this.multiplierRange = 20.0;
    
    // DOM elements
    this.elements = {
      multiplierInput: document.getElementById('multiplierInput'),
      addMultiplierBtn: document.getElementById('addMultiplier'),
      generatePredictionBtn: document.getElementById('generatePrediction'),
      multipliersContainer: document.getElementById('multipliersContainer'),
      multiplierCount: document.getElementById('multiplierCount'),
      noMultipliersMessage: document.getElementById('noMultipliersMessage'),
      deleteAllMultipliersBtn: document.getElementById('deleteAllMultipliers'),
      predictionResults: document.getElementById('predictionResults'),
      accuracyDisplay: document.getElementById('accuracyDisplay'),
      rangeDisplay: document.getElementById('rangeDisplay'),
      predictedValue: document.getElementById('predictedValue'),
      rangeLowerValue: document.getElementById('rangeLowerValue'),
      rangeUpperValue: document.getElementById('rangeUpperValue'),
      confidenceBar: document.getElementById('confidenceBar'),
      confidencePercentage: document.getElementById('confidencePercentage'),
      riskLevel: document.getElementById('riskLevel'),
      riskDisplay: document.getElementById('riskDisplay'),
      marketRegime: document.getElementById('marketRegime'),
      spikeStatus: document.getElementById('spikeStatus'),
      spikeMessage: document.getElementById('spikeMessage'),
      volatilityBar: document.getElementById('volatilityBar'),
      predictionTimestamp: document.getElementById('predictionTimestamp')
    };
  }
  
  initialize() {
    // Load saved data
    this.loadFromLocalStorage();
    
    // Setup event listeners
    this.setupEventListeners();
    
    // Update UI
    this.updateMultipliersDisplay();
    this.updateAccuracyDisplay();
    this.updateRangeDisplay();
    
    // Auto-scroll to bottom
    this.scrollToBottom();
    
    // Enable prediction button if we have enough multipliers
    this.elements.generatePredictionBtn.disabled = this.multipliers.length < 50;
  }
  
  setupEventListeners() {
    // Add multiplier button
    this.elements.addMultiplierBtn.addEventListener('click', () => {
      this.addMultiplier();
    });
    
    // Enter key in multiplier input
    this.elements.multiplierInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        this.addMultiplier();
      }
    });
    
    // Generate prediction button
    this.elements.generatePredictionBtn.addEventListener('click', () => {
      this.generatePrediction();
    });
    
    // Delete all multipliers button
    this.elements.deleteAllMultipliersBtn.addEventListener('click', () => {
      if (this.multipliers.length === 0) return;
      
      const deleteModal = document.getElementById('deleteAllModal');
      const deleteMessage = document.getElementById('deleteMessage');
      
      deleteMessage.textContent = 'Are you sure you want to delete all multipliers? This cannot be undone.';
      deleteModal.classList.add('active');
    });
  }
  
  addMultiplier() {
    const value = this.elements.multiplierInput.value.trim();
    if (value === '') return;
    
    const numericValue = parseFloat(value);
    if (isNaN(numericValue) || numericValue <= 0) {
      showSpeechBubble('Please enter a valid positive number');
      return;
    }
    
    this.multipliers.push(numericValue);
    this.updateMultipliersDisplay();
    this.saveToLocalStorage();
    
    // Enable prediction button if we have enough multipliers
    this.elements.generatePredictionBtn.disabled = this.multipliers.length < 50;
    
    // Clear input and focus
    this.elements.multiplierInput.value = '';
    this.elements.multiplierInput.focus();
    
    // Auto-scroll to bottom
    this.scrollToBottom();
  }
  
  deleteMultiplier(index) {
    if (index >= 0 && index < this.multipliers.length) {
      this.multipliers.splice(index, 1);
      this.updateMultipliersDisplay();
      this.saveToLocalStorage();
      
      // Disable prediction button if not enough multipliers
      this.elements.generatePredictionBtn.disabled = this.multipliers.length < 50;
    }
  }
  
  updateMultipliersDisplay() {
    // Update count
    this.elements.multiplierCount.textContent = this.multipliers.length;
    
    // Show/hide no multipliers message
    if (this.multipliers.length === 0) {
      this.elements.noMultipliersMessage.style.display = 'block';
    } else {
      this.elements.noMultipliersMessage.style.display = 'none';
    }
    
    // Clear container
    this.elements.multipliersContainer.innerHTML = '';
    
    // Add multipliers in reverse order (newest at top)
    for (let i = this.multipliers.length - 1; i >= 0; i--) {
      const index = i;
      const multiplier = this.multipliers[index];
      
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
        
        const deleteModal = document.getElementById('deleteModal');
        const deleteMessage = document.getElementById('deleteMessage');
        
        deleteMessage.textContent = `Are you sure you want to delete multiplier ${multiplier}?`;
        deleteModal.setAttribute('data-index', index);
        deleteModal.classList.add('active');
      });
      
      // Add to container
      this.elements.multipliersContainer.appendChild(multiplierItem);
    }
  }
  
  generatePrediction() {
    if (this.multipliers.length < 50) {
      showSpeechBubble(`Need ${50 - this.multipliers.length} more multipliers for prediction`);
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
    // Update predicted value
    this.elements.predictedValue.textContent = prediction.predictedValue.toFixed(2);
    
    // Update range values
    this.elements.rangeLowerValue.textContent = prediction.rangeLower.toFixed(2);
    this.elements.rangeUpperValue.textContent = prediction.rangeUpper.toFixed(2);
    
    // Update confidence
    const confidencePercent = Math.round(prediction.confidence * 100);
    this.elements.confidencePercentage.textContent = `${confidencePercent}%`;
    this.elements.confidenceBar.style.width = `${confidencePercent}%`;
    
    // Update risk assessment
    this.elements.riskLevel.textContent = prediction.riskLevel;
    this.elements.riskDisplay.textContent = prediction.riskLevel;
    
    // Update last prediction time
    this.elements.predictionTimestamp.textContent = 'Just now';
    
    // Re-enable button
    this.elements.generatePredictionBtn.disabled = false;
    this.elements.generatePredictionBtn.textContent = 'Generate Prediction';
    
    // Show prediction results with animation
    this.elements.predictionResults.classList.add('visible');
    
    // Show speech bubble with cash out range
    showSpeechBubble(`Prediction generated: ${prediction.predictedValue.toFixed(2)} | Cash-out range: ${prediction.rangeLower.toFixed(2)}-${prediction.rangeUpper.toFixed(2)}`);
  }
  
  updateAccuracyDisplay() {
    // Update accuracy display
    this.elements.accuracyDisplay.textContent = `${this.accuracy.toFixed(2)}%`;
    
    // Update accuracy circle
    document.getElementById('accuracyCircle').style.setProperty('--accuracy', `${this.accuracy}%`);
    document.getElementById('accuracyValue').textContent = `${Math.round(this.accuracy)}%`;
    
    // Update confidence meter
    document.getElementById('confidenceMeter').style.width = `${this.accuracy}%`;
  }
  
  updateRangeDisplay() {
    // Update range display
    this.elements.rangeDisplay.textContent = `┬▒${this.multiplierRange.toFixed(2)}%`;
    
    // Update volatility
    const volatilityPercent = Math.min(100, this.multiplierRange * 1.5);
    this.elements.volatilityBar.style.width = `${volatilityPercent}%`;
    document.getElementById('volatilityMeter').style.width = `${volatilityPercent}%`;
  }
  
  loadFromLocalStorage() {
    try {
      const savedMultipliers = localStorage.getItem('crashPredictor_multipliers');
      if (savedMultipliers) {
        this.multipliers = JSON.parse(savedMultipliers);
      }
    } catch (e) {
      console.error('Error loading from localStorage:', e);
    }
  }
  
  saveToLocalStorage() {
    try {
      localStorage.setItem('crashPredictor_multipliers', JSON.stringify(this.multipliers));
    } catch (e) {
      console.error('Error saving to localStorage:', e);
    }
  }
  
  scrollToBottom() {
    this.elements.multipliersContainer.scrollTop = this.elements.multipliersContainer.scrollHeight;
  }
}

/**
 * Voice Recognition Handler
 */
class VoiceRecognition {
  constructor(predictionEngine, uiManager) {
    this.predictionEngine = predictionEngine;
    this.uiManager = uiManager;
    this.speechRecognition = null;
    this.isListening = false;
    this.voiceToggle = document.getElementById('voiceToggle');
    this.voiceIndicator = document.getElementById('voiceIndicator');
    this.voiceCommands = document.getElementById('voiceCommands');
  }
  
  initialize() {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      this.voiceToggle.disabled = true;
      this.voiceToggle.parentElement.parentElement.innerHTML = 'Voice recognition is not supported in your browser.';
      showSpeechBubble('Voice recognition is not supported in your browser.');
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
      
      // Only change toggle if it's not supposed to be active
      if (!this.voiceToggle.checked) {
        this.voiceToggle.checked = false;
        localStorage.setItem('voiceRecognitionEnabled', 'false');
      }
      
      // Restart recognition if toggle is still on
      if (this.voiceToggle.checked) {
        try {
          this.speechRecognition.start();
        } catch (e) {
          console.error('Error restarting speech recognition:', e);
        }
      }
    };
    
    this.speechRecognition.onresult = (event) => {
      let transcript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          transcript += event.results[i][0].transcript;
        }
      }
      
      if (transcript.trim() !== '') {
        this.processVoiceCommand(transcript.toLowerCase());
      }
    };
    
    this.speechRecognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      if (event.error === 'not-allowed') {
        showSpeechBubble('Microphone access denied. Please enable microphone permissions.');
        this.voiceToggle.checked = false;
        localStorage.setItem('voiceRecognitionEnabled', 'false');
      }
    };
    
    // Voice toggle checkbox
    this.voiceToggle.addEventListener('change', () => {
      localStorage.setItem('voiceRecognitionEnabled', this.voiceToggle.checked);
      this.toggleVoiceRecognition();
    });
    
    // Set initial state
    if (localStorage.getItem('voiceRecognitionEnabled') === 'true') {
      this.voiceToggle.checked = true;
      try {
        this.speechRecognition.start();
        this.isListening = true;
        this.voiceIndicator.classList.add('active');
      } catch (e) {
        console.error('Error starting speech recognition on load:', e);
      }
    }
  }
  
  toggleVoiceRecognition() {
    if (!this.speechRecognition) {
      this.voiceToggle.checked = false;
      localStorage.setItem('voiceRecognitionEnabled', 'false');
      showSpeechBubble('Voice recognition is not supported in your browser.');
      return;
    }
    
    if (this.voiceToggle.checked) {
      try {
        this.speechRecognition.start();
        showSpeechBubble('Voice recognition activated');
      } catch (e) {
        console.error('Error starting speech recognition:', e);
        this.voiceToggle.checked = false;
        localStorage.setItem('voiceRecognitionEnabled', 'false');
        showSpeechBubble('Error starting voice recognition. Please try again.');
      }
    } else {
      try {
        this.speechRecognition.stop();
        showSpeechBubble('Voice recognition deactivated');
      } catch (e) {
        console.error('Error stopping speech recognition:', e);
      }
    }
  }
  
  processVoiceCommand(command) {
    console.log('Voice command:', command);
    
    // Add recent command to history
    const commands = this.voiceCommands.querySelectorAll('.voice-command');
    commands.forEach(cmd => cmd.classList.remove('recent'));
    
    const newCommand = document.createElement('div');
    newCommand.className = 'voice-command recent';
    newCommand.textContent = command;
    this.voiceCommands.insertBefore(newCommand, this.voiceCommands.firstChild);
    
    if (commands.length >= 3) {
      commands[commands.length - 1].remove();
    }
    
    // Process command
    if (command.includes('multiplier') || command.includes('multipliers')) {
      const numberMatch = command.match(/[\d.]+/);
      if (numberMatch) {
        const value = parseFloat(numberMatch[0]);
        if (!isNaN(value) && value > 0) {
          this.uiManager.multipliers.push(value);
          this.uiManager.updateMultipliersDisplay();
          this.uiManager.saveToLocalStorage();
          
          // Enable prediction button if we have enough multipliers
          this.uiManager.elements.generatePredictionBtn.disabled = this.uiManager.multipliers.length < 50;
          
          // Auto-scroll to bottom
          this.uiManager.scrollToBottom();
          
          showSpeechBubble(`Added multiplier ${value}`);
          
          // Auto-focus input and move cursor down
          setTimeout(() => {
            this.uiManager.elements.multiplierInput.focus();
            this.uiManager.elements.multiplierInput.value = '';
          }, 500);
        }
      }
    } else if (command.includes('delete last') || command.includes('remove last')) {
      if (this.uiManager.multipliers.length > 0) {
        const lastMultiplier = this.uiManager.multipliers[this.uiManager.multipliers.length - 1];
        this.uiManager.deleteMultiplier(this.uiManager.multipliers.length - 1);
        showSpeechBubble(`Deleted multiplier ${lastMultiplier}`);
      } else {
        showSpeechBubble('No multipliers to delete');
      }
    } else if (command.includes('delete all') || command.includes('clear all')) {
      if (this.uiManager.multipliers.length > 0) {
        const deleteModal = document.getElementById('deleteAllModal');
        const deleteMessage = document.getElementById('deleteMessage');
        
        deleteMessage.textContent = 'Are you sure you want to delete all multipliers? This cannot be undone.';
        deleteModal.classList.add('active');
        showSpeechBubble('Confirm deletion of all multipliers');
      } else {
        showSpeechBubble('No multipliers to delete');
      }
    } else if (command.includes('done') || command.includes('predict')) {
      if (this.uiManager.multipliers.length >= 50) {
        this.uiManager.generatePrediction();
        showSpeechBubble('Generating prediction with cash out range...');
      } else {
        showSpeechBubble(`Need ${50 - this.uiManager.multipliers.length} more multipliers for prediction`);
      }
    } else if (command.includes('accuracy') || command.includes('how accurate')) {
      showSpeechBubble(`Current accuracy is ${this.predictionEngine.accuracy.toFixed(2)}%`);
    } else if (command.includes('range') || command.includes('prediction range')) {
      showSpeechBubble(`Current prediction range is ┬▒${this.predictionEngine.multiplierRange.toFixed(2)}%`);
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
    this.jankDetected = false;
    this.jankCount = 0;
    this.jankThreshold = 30; // ms
  }
  
  startMonitoring() {
    // Start FPS monitoring
    this.lastFpsUpdate = performance.now();
    this.lastFrameTime = this.lastFpsUpdate;
    
    const monitorFrame = () => {
      const now = performance.now();
      this.frameCount++;
      
      // Calculate FPS
      if (now > this.lastFpsUpdate + 1000) {
        this.fps = Math.round((this.frameCount * 1000) / (now - this.lastFpsUpdate));
        this.fpsSamples.push(this.fps);
        
        // Keep only last 10 samples
        if (this.fpsSamples.length > 10) {
          this.fpsSamples.shift();
        }
        
        this.frameCount = 0;
        this.lastFpsUpdate = now;
      }
      
      // Check for jank
      const frameTime = now - this.lastFrameTime;
      if (frameTime > this.jankThreshold) {
        this.jankDetected = true;
        this.jankCount++;
      }
      
      this.lastFrameTime = now;
      
      // Continue monitoring
      requestAnimationFrame(monitorFrame);
    };
    
    requestAnimationFrame(monitorFrame);
    
    // Log performance stats periodically
    setInterval(() => {
      this.logPerformanceStats();
    }, 5000);
  }
  
  logPerformanceStats() {
    // Calculate average FPS
    const avgFps = this.fpsSamples.reduce((sum, val) => sum + val, 0) / this.fpsSamples.length;
    
    console.log(`📊 Performance: ${avgFps.toFixed(1)} FPS | Jank: ${this.jankCount} events`);
    
    // Reset counters
    this.jankCount = 0;
  }
}

/**
 * Global helper functions
 */
function showSpeechBubble(message) {
  const speechBubble = document.getElementById('speechBubble');
  if (!speechBubble) return;
  
  speechBubble.textContent = message;
  speechBubble.classList.add('visible');
  
  // Hide after 3 seconds
  setTimeout(() => {
    speechBubble.classList.remove('visible');
  }, 3000);
}

function fetchMetadata(predictionEngine) {
  // Try to load from GitHub
  fetch('https://raw.githubusercontent.com/eustancek/Crashpredictor/main/metadata.json')
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      // Update accuracy
      const accuracy = data.accuracy || 75.0;
      const multiplierRange = data.range_percentage || 20.0;
      
      predictionEngine.updateAccuracy(accuracy, multiplierRange);
      
      // Update UI
      const uiManager = window.uiManager;
      if (uiManager) {
        uiManager.accuracy = accuracy;
        uiManager.multiplierRange = multiplierRange;
        uiManager.updateAccuracyDisplay();
        uiManager.updateRangeDisplay();
      }
      
      // Update last updated time
      const lastUpdated = document.getElementById('lastUpdated');
      if (lastUpdated) {
        const now = new Date();
        lastUpdated.textContent = now.toLocaleTimeString([], { 
          hour: '2-digit', 
          minute: '2-digit'
        });
      }
    })
    .catch(error => {
      console.error('Error loading metadata:', error);
    });
}

// Make these functions available globally for event handlers
window.updateUIWithPrediction = function(prediction) {
  if (window.uiManager) {
    window.uiManager.updateUIWithPrediction(prediction);
  }
};

window.handlePredictionError = function(error) {
  showSpeechBubble(`Prediction error: ${error}`);
  if (window.uiManager && window.uiManager.elements) {
    window.uiManager.elements.generatePredictionBtn.disabled = false;
    window.uiManager.elements.generatePredictionBtn.textContent = 'Generate Prediction';
  }
};
