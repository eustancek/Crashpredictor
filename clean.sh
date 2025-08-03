#!/bin/bash
# clean.sh - Clean up old model versions (keep only latest_model.keras)

echo "Cleaning up old model versions..."

# Remove all model files except latest_model.keras
find models -type f \( -name "*.keras" -o -name "*.h5" -o -name "*.pkl" \) ! -name "latest_model.keras" -exec rm -f {} \;

echo "Old model versions cleaned up."