import React, { useEffect } from "react"
import {api} from  "../api/speectAPI.ts"
export default function mainPage() {
  const [data,setData] = React.useState(null);
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
  return (
    <div>
        <p>AQUI INTRODUCE LAS MAMADAS</p>
        <p>{data}</p>
    </div>
  )
}
