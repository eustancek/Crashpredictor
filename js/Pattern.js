// Advanced Pattern Detection
export function detectPatterns(data) {
    return {
        movingAverages: calculateMovingAverages(data),
        momentum: detectMomentum(data),
        timeBased: detectTimePatterns(data),
        bayesian: bayesianAnalysis(data),
        volatility: calculateVolatility(data)
    };
}

function calculateMovingAverages(data) {
    return {
        ma5: movingAverage(data, 5),
        ma10: movingAverage(data, 10),
        ma20: movingAverage(data, 20)
    };
}

function detectMomentum(data) {
    return {
        current: calculateMomentum(data),
        strength: calculateMomentumStrength(data)
    };
}

function detectTimePatterns(data) {
    return {
        daily: detectDailyPatterns(data),
        weekly: detectWeeklyPatterns(data)
    };
}