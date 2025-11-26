import React from 'react'
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import MainPage from './pages/mainPage';

function App(){
    return(
        <Router>
            <Routes>
                <Route path="/prendelatele" element={<MainPage />} />
            </Routes>
        </Router>
    )
}

export default App