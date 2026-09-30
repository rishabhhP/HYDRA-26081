import React from 'react';
import {createRoot} from 'react-dom/client';
import App from './App';
import './style.css';
class Boundary extends React.Component<{children:React.ReactNode},{error:boolean}>{
 state={error:false};
 static getDerivedStateFromError(){return {error:true};}
 render(){return this.state.error?<main className="fatal"><h1>Workspace could not load</h1><p>Reload the page to reconnect to HYDRA.</p><button onClick={()=>location.reload()}>Reload</button></main>:this.props.children;}
}
createRoot(document.getElementById('root')!).render(<Boundary><App/></Boundary>);
