import {useEffect, useMemo, useState} from 'react';
import {Activity, AlertTriangle, CalendarDays, CloudRain, Database, Gauge, Scale, Target} from 'lucide-react';
import {fmt, get, label} from './api';
import {ChartCard, TimeSeries} from './Charts';

type StateOption={name:string;code:string};
type Expert={name:string;value:number|null;weight:number|null};
type ReplayRow={issue_date:string;valid_date:string;actual_mm:number;hydra_mm:number;interval80:[number,number];spread_mm:number;experts:Expert[];
 p_rain?:number;wet_amount_mm?:number;p_heavy_area?:Record<string,number>;p_heavy_max_cell?:Record<string,number>;local_peak_mm?:number;actual_local_max_mm?:number;actual_heavy_area?:Record<string,number>;regime?:string;interval_stratum?:string};
type Contingency={events:number;recall:number|null;precision:number|null;csi:number|null;brier?:number|null;trustworthy:boolean;definition?:string};
type CellMetrics={heavy:Record<string,Contingency>;occurrence:Contingency&{accuracy:number|null};interval?:{coverage:number|null;mean_width:number|null}};
type StateMetrics={local_heavy:Record<string,Contingency>;local_peak:{max_observed:number|null;predicted_at_max:number|null;ratio_at_max:number|null}};
type Interval={nominal:number|null;label:string;method:string;legacy?:boolean};
type Dynamics={mean_std:number|null;mean_total_variation_step:number|null;mean_normalised_entropy:number|null;static_warning:boolean};
type Replay={status:string;state:string;model_version?:string;target?:string;unit?:string;lead_hours?:number;history_days?:number;training_cutoff?:string;calibration?:string;coverage?:{start:string;end:string;days:number};
 interval?:Interval;heavy_thresholds_mm?:number[];state_p95_mm?:number;experts?:string[];metrics?:{state_scale?:{lead1?:StateMetrics}|null;cell_scale?:CellMetrics|null}|null;gate_dynamics?:Dynamics|null;
 message?:string;grid_cell_count?:number;nearest_grid_fallback?:boolean;rows:ReplayRow[]};

const EXPERT_COLOURS=['#76a394','#8f9fba','#b199b9','#d26e55','#5fa8c9','#c9a3d9','#9cc16a','#e08fb0','#e9c979'];
const BASELINES=['climatology','persistence','recent3','anom_persistence'];
const STATE_HEAVY_MM=20;

const mean=(values:number[])=>values.length?values.reduce((sum,value)=>sum+value,0)/values.length:NaN;
function scores(rows:ReplayRow[]){
 const ok=rows.filter(row=>Number.isFinite(row.actual_mm)&&Number.isFinite(row.hydra_mm));
 const err=ok.map(row=>row.hydra_mm-row.actual_mm);
 const expert=(row:ReplayRow,name:string)=>row.experts.find(e=>e.name===name)?.value;
 const refMae=(name:string)=>{const e=ok.map(row=>{const v=expert(row,name);return v==null?NaN:v-row.actual_mm}).filter(Number.isFinite);return e.length?mean(e.map(Math.abs)):NaN};
 const mae=mean(err.map(Math.abs)),rmse=Math.sqrt(mean(err.map(e=>e*e)));
 const inside=ok.filter(row=>row.actual_mm>=row.interval80[0]&&row.actual_mm<=row.interval80[1]).length;
 const wetObs=ok.map(row=>row.actual_mm>=1),wetFc=ok.map(row=>row.hydra_mm>=1);
 const heavyObs=ok.map(row=>row.actual_mm>=STATE_HEAVY_MM),heavyFc=ok.map(row=>row.hydra_mm>=STATE_HEAVY_MM);
 const hits=heavyObs.filter((o,i)=>o&&heavyFc[i]).length,events=heavyObs.filter(Boolean).length,forecasts=heavyFc.filter(Boolean).length;
 const peak=ok.reduce((best,row)=>!best||row.actual_mm>best.actual_mm?row:best,null as ReplayRow|null);
 const top=[...ok].sort((a,b)=>b.actual_mm-a.actual_mm).slice(0,5);
 const persistence=refMae('persistence'),climatology=refMae('climatology');
 return {mae,rmse,bias:mean(err),coverage:ok.length?inside/ok.length:NaN,width:mean(ok.map(row=>row.interval80[1]-row.interval80[0])),
  rainAccuracy:ok.length?wetObs.filter((o,i)=>o===wetFc[i]).length/ok.length:NaN,
  heavyRecall:events?hits/events:NaN,heavyPrecision:forecasts?hits/forecasts:NaN,heavyEvents:events,
  peak,topRatio:top.length?top.reduce((s,r)=>s+r.hydra_mm,0)/Math.max(top.reduce((s,r)=>s+r.actual_mm,0),1e-9):NaN,
  skillPersistence:Number.isFinite(persistence)&&persistence>0?1-mae/persistence:NaN,skillClimatology:Number.isFinite(climatology)&&climatology>0?1-mae/climatology:NaN};
}
const pct=(v:number|null|undefined,n=0)=>v==null||!Number.isFinite(v)?'—':`${fmt(v*100,n)}%`;
const signed=(v:number|null|undefined,n=2)=>v==null||!Number.isFinite(v)?'—':`${v>0?'+':''}${fmt(v,n)}`;

function Metric({name,value,note,warn}:{name:string;value:string;note?:string;warn?:boolean}){
 return <div className={warn?'replay-metric warn':'replay-metric'}><dt>{name}</dt><dd>{value}</dd>{note&&<small>{note}</small>}</div>;
}

export default function RollingRainfall({selectedState,states}:{selectedState?:string;states:StateOption[]}){
 const [state,setState]=useState(selectedState||'Maharashtra'),[data,setData]=useState<Replay|null>(null),[error,setError]=useState(''),[showExperts,setShowExperts]=useState<boolean|null>(null);
 useEffect(()=>{if(selectedState)setState(selectedState)},[selectedState]);
 useEffect(()=>{const ac=new AbortController();setData(null);setError('');get<Replay>(`hydra-rolling-rainfall?state=${encodeURIComponent(state)}`,ac.signal).then(setData).catch(e=>{if(e.name!=='AbortError')setError(e.message)});return()=>ac.abort();},[state]);
 const rows=data?.rows||[];
 const expertNames=useMemo(()=>data?.experts||(rows[0]?.experts.map(e=>e.name))||BASELINES,[data,rows]);
 const v3=Boolean(rows[0]&&rows[0].p_rain!=null);
 const experts=useMemo(()=>expertNames.map((name,index)=>({name,index,values:rows.map(row=>row.experts.find(e=>e.name===name)?.value??null),weights:rows.map(row=>{const w=row.experts.find(e=>e.name===name)?.weight;return w==null?null:w*100})})),[rows,expertNames]);
 const s=useMemo(()=>scores(rows),[rows]);
 const averageSpread=rows.length?mean(rows.map(row=>row.spread_mm)):0;
 const weightStd=useMemo(()=>{if(!rows.length)return NaN;return mean(experts.map(e=>{const w=e.weights.filter((v):v is number=>v!=null);const m=mean(w);return Math.sqrt(mean(w.map(v=>(v-m)**2)))/100}))},[experts,rows]);
 if(error)return <p role="alert">{error}</p>;
 if(!data)return <p className="loading">Preparing the HYDRA rainfall replay…</p>;
 if(data.status!=='available')return <div className="empty"><Database size={22}/><h3>Rainfall replay unavailable</h3><p>{data.message}</p></div>;
 const interval=data.interval||{nominal:null,label:'Interval',method:'',legacy:true};
 const legacy=Boolean(interval.legacy);
 const nominal=interval.nominal??null;
 const coverageOff=nominal!=null?Math.abs(s.coverage-nominal)>0.05:true;
 const expertsVisible=showExperts??!v3;
 const cell=data.metrics?.cell_scale||null,local=data.metrics?.state_scale?.lead1||null;
 const staticGate=data.gate_dynamics?.static_warning??weightStd<0.02;
 return <div className="rolling-rainfall">
  <section className="replay-hero">
   <div><small>OVERVIEW · HYDRA ROLLING BACKTEST · {(data.model_version||'v2').toUpperCase()}</small><h3>Six-month rainfall replay</h3><p>{data.message}</p></div>
   <label className="replay-state"><span>STATE / UT</span><select aria-label="Rainfall replay state" value={state} onChange={event=>setState(event.target.value)}>{states.map(option=><option key={option.code} value={option.name}>{option.name}</option>)}</select></label>
  </section>
  {legacy&&<div className="replay-warning" role="note"><AlertTriangle size={15}/><p><b>Legacy replay.</b> This file predates HYDRA v3: four rainfall-only experts on state averages, a nearly static gate, and a band built from the p90 residual. That band covers {pct(s.coverage,1)} here, so it is not an 80% interval. Rebuild with <code>scripts/build_hydra_rolling_rainfall_replay.py</code> after staging ERA5.</p></div>}
  <div className="replay-facts">
   <div><CalendarDays size={15}/><span>Window</span><b>{data.coverage?.start} → {data.coverage?.end}</b></div>
   <div><Activity size={15}/><span>Issue rule</span><b>{data.history_days} prior days → +{data.lead_hours}h</b></div>
   <div><Scale size={15}/><span>HYDRA replay MAE</span><b>{fmt(s.mae)} {data.unit}</b></div>
   <div><Database size={15}/><span>State grid</span><b>{fmt(data.grid_cell_count,0)} ERA5 cells{v3?' · cell-level model':''}</b></div>
  </div>
  <ChartCard title={`${data.state} · daily rainfall`} subtitle={`Observed ERA5 state mean vs HYDRA’s +${data.lead_hours}h neural-gated blend · ${data.unit}`}>
   <TimeSeries labels={rows.map(row=>row.valid_date)} unit={data.unit} band={{lower:rows.map(row=>row.interval80[0]),upper:rows.map(row=>row.interval80[1]),label:interval.label,color:'#d6a85f'}} series={[
    {label:'Observed ERA5 rainfall',values:rows.map(row=>row.actual_mm),color:'#dbe4de',width:2.5},
    {label:'HYDRA adaptive blend',values:rows.map(row=>row.hydra_mm),color:'#d6a85f',width:2.2},
    ...(expertsVisible?experts.map(expert=>({label:label(expert.name),values:expert.values,color:EXPERT_COLOURS[expert.index%EXPERT_COLOURS.length],dash:'5 4',width:1.1})):[]),
   ]}/>
   <label className="replay-toggle"><input type="checkbox" checked={expertsVisible} onChange={event=>setShowExperts(event.target.checked)}/> Show the {experts.length} expert forecasts</label>
  </ChartCard>
  <section className="replay-scorecard" aria-label="Replay evaluation">
   <div className="replay-score-head"><Target size={14}/><h3>Evaluation · state mean</h3><span>{rows.length} days · lead +{data.lead_hours}h</span></div>
   <dl>
    <Metric name="MAE" value={`${fmt(s.mae)} ${data.unit}`}/>
    <Metric name="RMSE" value={`${fmt(s.rmse)} ${data.unit}`}/>
    <Metric name="Bias (forecast − observed)" value={`${signed(s.bias)} ${data.unit}`}/>
    <Metric name="Rain / no-rain accuracy" value={pct(s.rainAccuracy,1)} note="state mean ≥ 1 mm"/>
    <Metric name={`Heavy recall (≥ ${STATE_HEAVY_MM} mm)`} value={pct(s.heavyRecall)} note={`${s.heavyEvents} observed days`} warn={s.heavyRecall<0.5}/>
    <Metric name={`Heavy precision (≥ ${STATE_HEAVY_MM} mm)`} value={pct(s.heavyPrecision)}/>
    <Metric name="Peak day" value={s.peak?`${fmt(s.peak.hydra_mm,1)} vs ${fmt(s.peak.actual_mm,1)}`:'—'} note={s.peak?`${s.peak.valid_date} · forecast vs observed`:undefined} warn={s.peak?s.peak.hydra_mm<0.7*s.peak.actual_mm:false}/>
    <Metric name="Top-5 day ratio" value={fmt(s.topRatio,2)} note="Σ forecast / Σ observed on the 5 wettest days" warn={s.topRatio<0.7}/>
    <Metric name={`Interval coverage${nominal!=null?` (target ${pct(nominal)})`:''}`} value={pct(s.coverage,1)} note={`mean width ${fmt(s.width,1)} ${data.unit}`} warn={coverageOff}/>
    <Metric name="MAE skill vs persistence" value={signed(s.skillPersistence)} note="positive = HYDRA better" warn={s.skillPersistence<0}/>
    <Metric name="MAE skill vs climatology" value={signed(s.skillClimatology)} note="positive = HYDRA better" warn={s.skillClimatology<0}/>
    <Metric name="Gate weight variation" value={fmt(data.gate_dynamics?.mean_std??weightStd,3)} note="mean std of expert weights" warn={staticGate}/>
   </dl>
  </section>
  {v3&&<>
   <ChartCard title={`${data.state} · rain and heavy-rain probability`} subtitle="Share of the state's grid cells the model expects to be wet or heavy, and the highest single-cell heavy-rain probability · %">
    <TimeSeries labels={rows.map(row=>row.valid_date)} unit="%" series={[
     {label:'Expected wet area (P ≥ 1 mm)',values:rows.map(row=>row.p_rain==null?null:row.p_rain*100),color:'#5fa8c9',width:1.8},
     {label:'Expected area ≥ 20 mm',values:rows.map(row=>row.p_heavy_area?.['20']==null?null:row.p_heavy_area['20']*100),color:'#d6a85f',width:1.8},
     {label:'Max cell P(≥ 64.5 mm)',values:rows.map(row=>row.p_heavy_max_cell?.['64.5']==null?null:row.p_heavy_max_cell['64.5']*100),color:'#d26e55',width:1.8},
     {label:'Observed area ≥ 20 mm',values:rows.map(row=>row.actual_heavy_area?.['20']==null?null:row.actual_heavy_area['20']*100),color:'#dbe4de',dash:'3 3',width:1.3},
    ]}/>
   </ChartCard>
   <ChartCard title={`${data.state} · local extremes inside the state`} subtitle={`Largest grid-cell value, which the state mean hides · ${data.unit}`}>
    <TimeSeries labels={rows.map(row=>row.valid_date)} unit={data.unit} series={[
     {label:'Observed wettest cell',values:rows.map(row=>row.actual_local_max_mm),color:'#dbe4de',width:2.2},
     {label:'HYDRA wettest cell',values:rows.map(row=>row.local_peak_mm),color:'#d26e55',width:2}]}/>
   </ChartCard>
   {cell&&<section className="replay-scorecard" aria-label="Grid-cell evaluation">
    <div className="replay-score-head"><CloudRain size={14}/><h3>Evaluation · grid cells in {data.state}</h3><span>probability heads, decision at P ≥ 0.5</span></div>
    <dl>
     <Metric name="Rain occurrence accuracy" value={pct(cell.occurrence.accuracy,1)} note={`Brier ${fmt(cell.occurrence.brier,3)}`}/>
     {(data.heavy_thresholds_mm||[20,64.5]).map(t=>{const h=cell.heavy[String(t)];return h?<Metric key={t} name={`Heavy ≥ ${t} mm recall / precision`} value={`${pct(h.recall)} / ${pct(h.precision)}`} note={`${h.events} cell-days${h.trustworthy?'':' · too few to trust'}`} warn={h.recall!=null&&h.recall<0.5}/>:null})}
     {cell.heavy.state_p95&&<Metric name={`Heavy ≥ state p95 (${fmt(data.state_p95_mm,1)} mm)`} value={`${pct(cell.heavy.state_p95.recall)} / ${pct(cell.heavy.state_p95.precision)}`} note="recall / precision"/>}
     {local&&Object.entries(local.local_heavy).map(([t,h])=><Metric key={t} name={`Any cell ≥ ${t} mm (state-day)`} value={`${pct(h.recall)} / ${pct(h.precision)}`} note={`${h.events} days · recall / precision`}/>)}
     {cell.interval&&<Metric name="Cell interval coverage" value={pct(cell.interval.coverage,1)} note={`width ${fmt(cell.interval.mean_width,1)} ${data.unit}`}/>}
     {local&&<Metric name="Wettest cell, peak day" value={`${fmt(local.local_peak.predicted_at_max,1)} vs ${fmt(local.local_peak.max_observed,1)}`} note="forecast vs observed"/>}
    </dl>
   </section>}
  </>}
  <ChartCard title={`${data.state} · neural gate allocation`} subtitle={`Expert weight per day · % · ${staticGate?'nearly static: the gate is not responding to conditions':'varies with the atmospheric state'}`}>
   <TimeSeries labels={rows.map(row=>row.valid_date)} unit="%" series={experts.map(expert=>({label:label(expert.name),values:expert.weights,color:EXPERT_COLOURS[expert.index%EXPERT_COLOURS.length],width:1.5}))}/>
  </ChartCard>
  <section className="replay-method">
   <div><small>HOW TO READ THIS</small><h3>Forecast first. Observed data second.</h3><p>For every date, HYDRA issued its rainfall value using data available on the issue day only. The observed ERA5 rainfall shown in white was held back until comparison. {v3?'HYDRA v3 predicts every 0.25° grid cell from ERA5 atmospheric predictors (CAPE, moisture, pressure, wind, cloud, radiation) and rainfall history, blends nine experts with a neural gate, and only then averages to the state.':'The coloured lines are the four rainfall-derived experts supplied to the legacy gate; the gold line is its weighted blend.'}</p></div>
   <dl><div><dt>Training cutoff</dt><dd>{data.training_cutoff}</dd></div><div><dt>Mean expert spread</dt><dd>{fmt(averageSpread)} {data.unit}</dd></div><div><dt>Replay RMSE</dt><dd>{fmt(s.rmse)} {data.unit}</dd></div><div><dt>Interval</dt><dd>{interval.label}</dd></div></dl>
  </section>
  <p className="footnote"><Gauge size={11}/> {legacy?interval.method:data.calibration} Expert spread measures disagreement between HYDRA’s inputs; it is not a confidence percentage. State-mean heavy thresholds ({STATE_HEAVY_MM} mm/day) are much stricter than cell thresholds because averaging smooths local downpours. No provider forecast is used in this replay.</p>
 </div>;
}
