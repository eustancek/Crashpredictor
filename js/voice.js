const recognition = new webkitSpeechRecognition();
recognition.continuous = true;
recognition.interimResults = false;
recognition.lang = 'en-US';

let currentInput = 'multiplier';

document.getElementById('toggle-voice').addEventListener('click', () => {
    if (recognition.running) {
        recognition.stop();
        document.getElementById('voice-status').textContent = 'Voice: OFF';
    } else {
        recognition.start();
        document.getElementById('voice-status').textContent = 'Voice: ON';
    }
});

recognition.onresult = function(event) {
    const transcript = event.results[event.resultIndex][0].transcript.toLowerCase();
    
    if (transcript.includes('actual')) {
        currentInput = 'crash';
        document.querySelector(`#${currentInput}`).focus();
    } else if (transcript.includes('multiplier')) {
        currentInput = 'multiplier';
        document.querySelector(`#${currentInput}`).focus();
    } else {
        const numberMatch = transcript.match(/(\d+(\.\d+)?)/);
        if (numberMatch) {
            const value = parseFloat(numberMatch[0]);
            const textarea = document.querySelector(`#${currentInput}`);
            textarea.value += value + '\n';
            textarea.scrollTop = textarea.scrollHeight;
            
            const lines = textarea.value.split('\n');
            if (lines.length > 5) {
                textarea.setSelectionRange(lines.slice(0, -2).join('\n').length + 1, lines.slice(0, -2).join('\n').length + 1);
            }
        }
    }
};