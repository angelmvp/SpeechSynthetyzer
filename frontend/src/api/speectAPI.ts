
const API_URL = "http://127.0.0.1:8000";

export const api = {
    get: async (endpoint: string) => {
        try {
            const complete_URL = `${API_URL}${endpoint}`;
            console.log("Fetching:", complete_URL);
            const response = await fetch(complete_URL, {
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            if (!response.ok) {
                const text = await response.text();
                throw new Error(`HTTP ${response.status} - ${text}`);
            }
            return await response.json();
        } catch (error: any) {
            console.error('Fetch error:', error);
            return { error: error.message || 'Error fetching data' };
        }
    },
    post: async (endpoint: string, data: any) => {
        try {
            console.log("Posting to:", `${API_URL}${endpoint}`, "with data:", data);
            const response = await fetch(`${API_URL}${endpoint}`, {
                method: 'POST',
                body: JSON.stringify(data),
                
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            if (!response.ok) {
                const text = await response.text();
                throw new Error(`HTTP ${response.status} - ${text}`);
            }
            return await response.json();
        } catch (error: any) {
            console.error('Post error:', error);
            return { error: error.message || 'Error posting data' };
        }
    }
}