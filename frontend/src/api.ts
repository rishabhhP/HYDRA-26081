import type {Selection} from './types';
export async function get<T>(path:string, signal?:AbortSignal):Promise<T>{
  const res=await fetch('/api/'+path,{signal});
  if(!res.ok)throw new Error((await res.json()).detail||`API request failed (${res.status})`);
  return res.json();
}
export async function post<T>(path:string,body:unknown):Promise<T>{
  const res=await fetch('/api/'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!res.ok)throw new Error((await res.json()).detail||`Request failed (${res.status})`);
  return res.json();
}
export const query=(s:Selection)=>new URLSearchParams(Object.entries(s).map(([k,v])=>[k,String(v)])).toString();
export const fmt=(x:number|null|undefined,n=2)=>x==null?'—':x.toLocaleString(undefined,{maximumFractionDigits:n});
export const label=(s:string)=>({tp_mm:'Rainfall',t2m_C_mean:'Temperature',wind_speed_mean:'Wind speed',anom_persistence:'Anomaly persistence',recent3:'Recent 3-day mean',gating_adaptive:'HYDRA adaptive',lgbm_season:'Seasonal LightGBM'}[s]||s.replaceAll('_',' '));
