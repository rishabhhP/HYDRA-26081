import {useEffect, useState} from 'react';
import {AlertTriangle, BadgeCheck, CloudRain, ShieldAlert, ShieldCheck} from 'lucide-react';
import {fmt, get} from './api';
import {ChartCard, TimeSeries} from './Charts';
import './postprocessed.css';

type Probs={state_10:number|null;state_20:number|null;'state_64.5':number|null;'local_64.5':number|null};
type Row={valid_date:string;observed_mm:number|null;observed_local_max_mm?:number|null;hydra_v3_mm:number|null;amount_mm:number|null;
 interval80:[number|null,number|null];v3_interval80?:[number|null,number|null];probabilities:Probs;alert:{state_20:string|null;'local_64.5':string|null};
 observed_events:Record<string,boolean|null>};
type Check={check:string;passed:boolean;required:boolean;detail:string};
type Gate={status:'validated'|'provisional'|'not_evaluated';failed:string[];checks:Check[];truth?:string;test_period?:[string,string]};
type Tier={tier:string;min_prob:number;issued:number;hit_rate:number|null;events_captured:number|null;events:number};
type Card={test_state_days:number;heavy_20mm_events:number;mae_mm:number;raw_mae_mm:number;bias_mm:number;mae_skill_vs_persistence:number;reliability:'good'|'fair'|'weak';heavy_rain_evidence:string};
type Payload={status:string;message?:string;state:string;model_version:string;base_model?:string;truth?:string;events:Record<string,string>;notes:string[];
 provenance:{kind:'out_of_sample'|'fitted';file:string;note:string};reliability?:Card|null;release_gate:Gate;tier_summary?:Record<string,Tier[]>|null;
 recent_alerts:Row[];rows:Row[]};

const pct=(v:number|null|undefined,n=0)=>v==null||!Number.isFinite(v)?'—':`${fmt(v*100,n)}%`;
const TIER_LABEL:Record<string,string>={watch:'Watch',alert:'Alert',warning:'Warning'};

function TierBadge({tier}:{tier:string|null}){
 if(!tier)return <span className="pp-tier none">none</span>;
 return <span className={`pp-tier ${tier}`}>{TIER_LABEL[tier]||tier}</span>;
}

function GateBadge({gate}:{gate:Gate}){
 const ok=gate.status==='validated';
 return <div className={`pp-gate ${gate.status}`} role="status">
  {ok?<ShieldCheck size={15}/>:<ShieldAlert size={15}/>}
  <div><b>{ok?'Validated release':gate.status==='provisional'?'Provisional: not all release checks pass':'Not evaluated yet'}</b>
   <small>{gate.truth?`Scored against ${gate.truth.toUpperCase()} observations`:''}{gate.test_period?` · ${gate.test_period[0]} to ${gate.test_period[1]}`:''}</small>
   {!ok&&gate.failed.length>0&&<small>Failing: {gate.failed.join('; ')}</small>}
  </div>
  <details><summary>Checks</summary><ul>{gate.checks.map(c=><li key={c.check} className={c.passed?'pass':c.required?'fail':'warn'}>{c.passed?'✓':c.required?'✗':'!'} {c.check}: {c.detail}</li>)}</ul></details>
 </div>;
}

function ReliabilityCard({card,state}:{card:Card;state:string}){
 return <section className={`pp-card ${card.reliability}`}>
  <div className="pp-card-head"><BadgeCheck size={14}/><h4>{state} reliability: {card.reliability}</h4></div>
  <dl>
   <div><dt>Test days</dt><dd>{card.test_state_days}</dd></div>
   <div><dt>Heavy days (≥20 mm)</dt><dd>{card.heavy_20mm_events}</dd></div>
   <div><dt>MAE (calibrated vs raw)</dt><dd>{fmt(card.mae_mm,2)} vs {fmt(card.raw_mae_mm,2)} mm</dd></div>
   <div><dt>Skill vs persistence</dt><dd>{pct(card.mae_skill_vs_persistence,1)}</dd></div>
   <div><dt>Bias</dt><dd>{fmt(card.bias_mm,2)} mm/day</dd></div>
   <div><dt>Heavy-rain evidence</dt><dd>{card.heavy_rain_evidence}</dd></div>
  </dl>
 </section>;
}

export default function PostprocessedPanel({state,unit='mm/day'}:{state:string;unit?:string}){
 const [data,setData]=useState<Payload|null>(null),[error,setError]=useState('');
 useEffect(()=>{const ac=new AbortController();setData(null);setError('');get<Payload>(`hydra-postprocessed?state=${encodeURIComponent(state)}`,ac.signal).then(setData).catch(e=>{if(e.name!=='AbortError')setError(e.message)});return()=>ac.abort();},[state]);
 if(error)return <p role="alert">{error}</p>;
 if(!data)return <p className="loading">Loading calibrated heavy-rain risk…</p>;
 if(data.status!=='available')return <div className="pp-empty"><CloudRain size={18}/><p>{data.message}</p></div>;
 const rows=data.rows;
 const labels=rows.map(r=>r.valid_date);
 const fitted=data.provenance.kind==='fitted';
 return <div className="pp-panel">
  <section className="pp-head">
   <div><small>HYDRA v3.4 · IMD-TARGET · POST-PROCESSED</small><h3>Calibrated rainfall and heavy-rain risk</h3>
    <p>Amounts re-weighted by recent expert skill, an 80% range with balanced tails, and heavy rain shown as probabilities with alert tiers. {data.provenance.note}</p></div>
   <GateBadge gate={data.release_gate}/>
  </section>
  {fitted&&<div className="pp-warn"><AlertTriangle size={14}/><p>These rows are in-sample fits, not forecasts. Run <code>python -m hydra_post.evaluate</code> to publish out-of-sample replay rows.</p></div>}
  {data.reliability&&<ReliabilityCard card={data.reliability} state={data.state}/>}
  <ChartCard title={`${data.state} · v3.4 rainfall amount`} subtitle={`Observed ${data.truth?.toUpperCase()||'IMD'} rainfall vs raw and calibrated HYDRA, with the adaptive 80% range · ${unit}`}>
   <TimeSeries labels={labels} unit={unit} band={{lower:rows.map(r=>r.interval80[0]),upper:rows.map(r=>r.interval80[1]),label:'Adaptive 80% range',color:'#5fa8c9'}} series={[
    {label:'Observed',values:rows.map(r=>r.observed_mm),color:'#dbe4de',width:2.4},
    {label:'HYDRA v3.4 calibrated',values:rows.map(r=>r.amount_mm),color:'#5fa8c9',width:2.1},
    {label:'HYDRA v3.4 raw',values:rows.map(r=>r.hydra_v3_mm),color:'#d6a85f',dash:'5 4',width:1.3}]}/>
  </ChartCard>
  <ChartCard title={`${data.state} · heavy-rain probability`} subtitle="Calibrated chance of each event · %. Tiers: watch ≥ 20%, alert ≥ 40%, warning ≥ 60%. Dashed lines mark days the event happened.">
   <TimeSeries labels={labels} unit="%" series={[
    {label:'P(state average ≥ 20 mm)',values:rows.map(r=>r.probabilities.state_20==null?null:r.probabilities.state_20*100),color:'#d6a85f',width:2},
    {label:'P(≥ 64.5 mm somewhere in state)',values:rows.map(r=>r.probabilities['local_64.5']==null?null:r.probabilities['local_64.5']*100),color:'#d26e55',width:2},
    {label:'Happened: state ≥ 20 mm',values:rows.map(r=>r.observed_events.state_20==null?null:r.observed_events.state_20?100:0),color:'#dbe4de',dash:'2 3',width:1},
    {label:'Happened: a cell ≥ 64.5 mm',values:rows.map(r=>r.observed_events['local_64.5']==null?null:r.observed_events['local_64.5']?95:0),color:'#9a6b5f',dash:'2 3',width:1}]}/>
  </ChartCard>
  {data.tier_summary&&<section className="pp-tiers">
   <h4>How the alert tiers performed here (out-of-sample)</h4>
   {(['state_20','local_64.5'] as const).map(ev=><div key={ev}>
    <p>{data.events[ev]} · {data.tier_summary![ev][0]?.events ?? 0} observed days</p>
    <table><thead><tr><th>Tier</th><th>Issued</th><th>Right when issued</th><th>Events caught</th></tr></thead>
     <tbody>{data.tier_summary![ev].map(t=><tr key={t.tier}><td><TierBadge tier={t.tier}/></td><td>{t.issued}</td><td>{pct(t.hit_rate)}</td><td>{pct(t.events_captured)}</td></tr>)}</tbody></table>
   </div>)}
  </section>}
  {data.recent_alerts.length>0&&<section className="pp-alerts">
   <h4>Latest alert and warning days</h4>
   <table><thead><tr><th>Date</th><th>State ≥ 20 mm</th><th>≥ 64.5 mm somewhere</th><th>v3.4 amount</th><th>Observed</th><th>Wettest cell</th></tr></thead>
    <tbody>{data.recent_alerts.slice().reverse().map(r=><tr key={r.valid_date}>
     <td>{r.valid_date}</td><td><TierBadge tier={r.alert.state_20}/> {pct(r.probabilities.state_20)}</td><td><TierBadge tier={r.alert['local_64.5']}/> {pct(r.probabilities['local_64.5'])}</td>
     <td>{fmt(r.amount_mm,1)} mm</td><td>{fmt(r.observed_mm,1)} mm</td><td>{r.observed_local_max_mm==null?'—':`${fmt(r.observed_local_max_mm,0)} mm`}</td></tr>)}</tbody></table>
  </section>}
  <ul className="pp-notes">{data.notes.map(n=><li key={n}>{n}</li>)}</ul>
 </div>;
}
