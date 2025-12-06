import React, { useEffect } from "react"
import {api} from  "../api/speectAPI.ts"
export default function mainPage() {
  const [data,setData] = React.useState(null);
  const [token,setToken] = React.useState(null);
  const [text,setText] = React.useState("");
  const [audioUrl,setAudioUrl] = React.useState("");
  useEffect(() => {
    const fetchData = async () => {
      const result = await api.get("/data");
      console.log('API result:', result);
      if (result && result.error) {
        // handle error gracefully
        setData(`Error: ${result.error}`);
      } else {
        setData(result?.data ?? JSON.stringify(result));
      }
    };
    fetchData();
  }, []);


  const sentText = async () => {
    const result = await api.post("/predict", { text: text });
    console.log('API result:', result);

    if (result && result.error) {
      setData(`Error: ${result.error}`);
      return;
    }

    // Expecting { token: { text, phonemes, ... }, audio: { base64, mime, filename } }
    const token = result?.token;
    if (!token) {
      setData("No token received");
      return;
    }

    setToken(token);
    setData(null);

    const audio = result?.audio;
    if (audio?.base64 && audio?.mime) {
      const byteCharacters = atob(audio.base64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: audio.mime });
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
    } else {
      setAudioUrl("");
    }
  }
  const showAudio = async () =>{
    const audio = new Audio()
    audio.src = audioUrl
    await audio.play()

  }


  return (
    <div>
      <h2>TEXT TO SPEECH BY MVP Angel Martin vazquez Perez</h2>
      <input type="text" value={text} onChange={e => setText(e.target.value)} />
      <button onClick={sentText}>Send Text</button>
      {data && <p>{data}</p>}

      {token && (
        <div style={{ marginTop: "16px" }}>
          <h3>Token recibido</h3>
          <p><strong>Texto:</strong> {text}</p>
          {Array.isArray(token.tokens) && Array.isArray(token.fonos) ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {token.tokens.map((tkn, idx) => (
                <div key={idx} style={{ padding: "8px", border: "1px solid #ccc", borderRadius: "6px" }}>
                  <div><strong>Token:</strong> {tkn}</div>
                  <div><strong>Fonemas:</strong> {Array.isArray(token.fonos[idx]) ? token.fonos[idx].join(" ") : token.fonos[idx]}</div>
                </div>
              ))}
            </div>
          ) : (
            <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(token, null, 2)}</pre>
          )}
        </div>
      )}

      {audioUrl && (
        <div style={{ marginTop: "16px" }}>
          <h3>Audio generado</h3>
          <audio src={audioUrl} controls />
        </div>
      )}
    </div>
  )
}
