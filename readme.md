# 🧠 Crash Predictor (Optimized for Speed)

Blazing fast crash prediction model with **sub-100ms response times** and **minimal cold starts**.

## ✨ Features

- ⚡ **Near-instant predictions** (typically 50-150ms)
- 🔥 **No cold starts** (model stays warm with keep-alive system)
- 📈 **Continual learning** (accuracy improves over time)
- 📏 **Dynamic prediction range** (expands as accuracy improves)
- 🧠 **Sophisticated pattern detection** (trends, volatility, spikes)

## 🚀 How It Works

This model uses a hybrid architecture (CNN + LSTM) optimized specifically for:
- Fast loading times (minimal dependencies)
- Quick inference (lightweight model structure)
- Persistent knowledge (accuracy improves over time)

## 🛠️ Technical Details

- **Framework**: TensorFlow (CPU-optimized)
- **Inference Engine**: ONNX Runtime for 2-3x faster predictions
- **Cold Start Prevention**: 
  - Automatic warmup on startup
  - 5-minute keep-alive pings
  - Minimal memory footprint

## 📊 Performance

| Metric | Value |
|--------|-------|
| Cold Start Time | < 3 seconds |
| First Prediction | < 500ms |
| Subsequent Predictions | 50-150ms |
| Memory Usage | < 300MB |

## 💻 API Documentation

### POST `/predict`
Make a crash prediction

**Request:**
```json
{
  "values": [1.2, 1.5, 1.8, ..., 9.7]  // 50 most recent crash values
}
