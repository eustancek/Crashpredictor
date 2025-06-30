// Chart Visualization
let accuracyChart = null;
let confidenceChart = null;

export function initCharts() {
    const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
    const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');

    accuracyChart = new Chart(accuracyCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Accuracy',
                data: [],
                borderColor: '#4CAF50',
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { min: 75, max: 100 }
            }
        }
    });

    confidenceChart = new Chart(confidenceCtx, {
        type: 'bar',
        data: {
            labels: ['Pattern Recognition', 'Volatility', 'Momentum'],
            datasets: [{
                label: 'Confidence Level',
                data: [0, 0, 0],
                backgroundColor: ['#2196F3', '#FF9800', '#9C27B0']
            }]
        }
    });
}

export function updateAccuracyChart(accuracy) {
    const now = new Date().toLocaleTimeString();
    accuracyChart.data.labels.push(now);
    accuracyChart.data.datasets[0].data.push(accuracy);
    accuracyChart.update();
}

export function updateConfidenceChart(patterns) {
    const patternData = [
        patterns.momentum.confidence,
        patterns.volatility.confidence,
        patterns.meanReversion.confidence
    ];
    confidenceChart.data.datasets[0].data = patternData;
    confidenceChart.update();
}