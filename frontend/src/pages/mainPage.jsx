import React, { useEffect } from "react"
import {api} from  "../api/speectAPI.ts"
export default function mainPage() {
  const [data,setData] = React.useState(null);
  const [tokens,setTokens] = React.useState([]);
  const [text,setText] = React.useState("");
  const [audioUrl,setAudioUrl] = React.useState("");
  const [audioProsodiaUrl,setAudioProsodiaUrl] = React.useState("");
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

    // Expecting { tokens: {token_text: {...}}, audio, audio_con_prosodia }
    const tokensObj = result?.tokens;
    if (!tokensObj || typeof tokensObj !== "object") {
      setData("No tokens received");
      setTokens([]);
      return;
    }

    const entries = Object.entries(tokensObj).map(([key, value]) => ({
      id: key,
      ...(value || {}),
    }));
    setTokens(entries);
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

    const audioProsodia = result?.audio_con_prosodia;
    if (audioProsodia?.base64 && audioProsodia?.mime) {
      const byteCharacters = atob(audioProsodia.base64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: audioProsodia.mime });
      const url = URL.createObjectURL(blob);
      setAudioProsodiaUrl(url);
    } else {
      setAudioProsodiaUrl("");
    }
  }
  const showAudio = async () =>{
    const audio = new Audio()
    audio.src = audioUrl
    await audio.play()

  }


  return (
    <div>
            {audioUrl && (
        <div style={{ marginTop: "16px" }}>
          <h3>Audio generado</h3>
          <audio src={audioUrl} controls />
        </div>
      )}

      {audioProsodiaUrl && (
        <div style={{ marginTop: "16px" }}>
          <h3>Audio con prosodia</h3>
          <audio src={audioProsodiaUrl} controls />
        </div>
      )}
      <h2>TEXT TO SPEECH BY MVP Angel Martin vazquez Perez</h2>
      <input type="text" value={text} onChange={e => setText(e.target.value)} />
      <button onClick={sentText}>Send Text</button>
      {data && <p>{data}</p>}

      {tokens.length > 0 && (
        <div style={{ marginTop: "16px", display: "grid", gap: "10px" }}>
          <h3>Tokens recibidos</h3>
          {tokens.map((tkn, idx) => (
            <div key={`${tkn.id}-${idx}`} style={{ border: "1px solid #ccc", borderRadius: "6px", padding: "8px" }}>
              <div><strong>Token:</strong> {tkn.token ?? tkn.id}</div>
              {!tkn.signo ? (
              <div>
                <div><strong>Fonos:</strong> {Array.isArray(tkn.fonos) ? tkn.fonos.join(" ") : JSON.stringify(tkn.fonos)}</div>
                <div><strong>Stress fono:</strong> {tkn.stress_fono ?? "N/A"}</div>
                <div><strong>Stress prosodia:</strong> {tkn.stress_prosodia ?? "N/A"}</div>
                <div><strong>Fonos prosodia:</strong> {Array.isArray(tkn.fonos_prosodia) ? tkn.fonos_prosodia.join(" ") : JSON.stringify(tkn.fonos_prosodia)}</div>
              </div>
              ) : null}
              
            </div>
          ))}
        </div>
      )}


    </div>
  )
}
