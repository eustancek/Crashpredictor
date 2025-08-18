// File: /api/predict/edge.js
export const config = {
  runtime: 'edge',
  regions: ['iad1'],  // Virginia region (optimal for US-East)
};

// Quantum-enhanced Hugging Face prediction
export default async function handler(request) {
  try {
    // Validate request method
    if (request.method !== 'POST') {
      return new Response(JSON.stringify({ error: 'Quantum entanglement requires POST requests' }), {
        status: 405,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Parse request body
    const requestData = await request.json();
    
    // Quantum data validation
    if (!requestData?.inputs || !Array.isArray(requestData.inputs)) {
      return new Response(JSON.stringify({ error: 'Quantum state requires "inputs" array' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Get secrets from environment (using your specified names)
    const HF_TOKEN = process.env.HUGGINGTOKEN;
    const HF_API = process.env.HUGGINGAPL;

    if (!HF_TOKEN || !HF_API) {
      throw new Error('Quantum entanglement failed - Missing API credentials');
    }

    // Quantum-enhanced fetch with timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 9000);  // 9s timeout

    const hfResponse = await fetch(HF_API, {
      signal: controller.signal,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${HF_TOKEN}`,
        'X-Quantum-Entanglement': 'v1.2'  // Special quantum header
      },
      body: JSON.stringify({
        inputs: requestData.inputs,
        parameters: {
          max_length: 128,
          quantum_fluctuation: 0.7,
          ...requestData.parameters
        }
      })
    });

    clearTimeout(timeoutId);

    // Handle Hugging Face errors
    if (!hfResponse.ok) {
      const errorData = await hfResponse.text();
      return new Response(JSON.stringify({
        error: `Quantum collapse occurred (${hfResponse.status})`,
        details: errorData.slice(0, 200) + (errorData.length > 200 ? '...' : '')
      }), {
        status: 502,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Process quantum results
    const result = await hfResponse.json();
    
    // Quantum compression
    const compressedResult = result[0]?.generated_text 
      ? { prediction: result[0].generated_text.trim() }
      : { predictions: result };

    // Return with quantum headers
    return new Response(JSON.stringify(compressedResult), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'X-Quantum-Entanglement': 'success',
        'Cache-Control': 'public, s-maxage=60, stale-while-revalidate=30'
      }
    });

  } catch (error) {
    // Quantum error handling
    console.error('Quantum decoherence:', error);
    return new Response(JSON.stringify({
      error: 'Space-time disruption detected',
      message: error.message
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}
