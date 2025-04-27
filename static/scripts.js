document.getElementById('toggle-dark-mode').addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
});

document.getElementById('generate-btn').addEventListener('click', async () => {
    const prompt = document.getElementById('prompt').value;
    const length = document.getElementById('length').value;
    const temperature = document.getElementById('temperature').value;
    const k = document.getElementById('k').value;

    if (!prompt.trim()) {
        alert('Prompt cannot be empty!');
        return;
    }

    const chatLog = document.getElementById('chat-log');
    const userMessage = document.createElement('div');
    userMessage.textContent = `You: ${prompt}`;
    chatLog.appendChild(userMessage);

    const loadingMessage = document.createElement('div');
    loadingMessage.textContent = 'Generating...';
    chatLog.appendChild(loadingMessage);

    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, length, temperature, k }),
        });

        const data = await response.json();
        chatLog.removeChild(loadingMessage);

        if (data.error) {
            const errorMessage = document.createElement('div');
            errorMessage.textContent = `Error: ${data.error}`;
            chatLog.appendChild(errorMessage);
        } else {
            const botMessage = document.createElement('div');
            botMessage.textContent = `Bot: ${data.generated_text}`;
            chatLog.appendChild(botMessage);
        }
    } catch (error) {
        chatLog.removeChild(loadingMessage);
        const errorMessage = document.createElement('div');
        errorMessage.textContent = `Error: ${error.message}`;
        chatLog.appendChild(errorMessage);
    }
});