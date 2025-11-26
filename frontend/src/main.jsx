import React, { StrictMode } from 'react'
import {createRoot} from 'react-dom/client'
import App from './app.jsx'
import NavBar from './components/navBar.jsx'
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <NavBar />
    <App />
  </StrictMode>
)