import {useEffect, useMemo, useState} from 'react';
import {Activity, CalendarDays, Database, Scale} from 'lucide-react';
import {fmt, get, label} from './api';
import {ChartCard, TimeSeries} from './Charts';

type StateOption={name:string;code:string};
type Expert={name:string;value:number;weight:number};
type ReplayRow={issue_date:string;valid_date:string;actual_mm:number;hydra_mm:number;interval80:[number,number];spread_mm:number;experts:Expert[]};
type Replay={status:string;state:string;target?:string;unit?:string;lead_hours?:number;history_days?:number;training_cutoff?:string;calibration?:string;coverage?:{start:string;end:string;days:number};message?:string;grid_cell_count?:number;nearest_grid_fallback?:boolean;rows:ReplayRow[]};

function metric(rows:ReplayRow[], select:(row:ReplayRow)=>number){
 if(!rows.length)return {mae:0,rmse:0};
 const errors=rows.map(row=>select(row)-row.actual_mm);
 return {mae:errors.reduce((sum,value)=>sum+Math.abs(value),0)/errors.length,rmse:Math.sqrt(errors.reduce((sum,value)=>sum+value*value,0)/errors.length)};
}

export default function RollingRainfall({selectedState,states}:{selectedState?:string;states:StateOption[]}){
 const [state,setState]=useState(selectedState||'Maharashtra'),[data,setData]=useState<Replay|null>(null),[error,setError]=useState('');
 useEffect(()=>{if(selectedState)setState(selectedState)},[selectedState]);
 useEffect(()=>{const ac=new AbortController();setData(null);setError('');get<Replay>(`hydra-rolling-rainfall?state=${encodeURIComponent(state)}`,ac.signal).then(setData).catch(e=>{if(e.name!=='AbortError')setError(e.message)});return()=>ac.abort();},[state]);
 const rows=data?.rows||[];
 const experts=useMemo(()=>['climatology','persistence','recent3','anom_persistence'].map((name,index)=>({name,index,values:rows.map(row=>row.experts.find(expert=>expert.name===name)?.value??null)})),[rows]);
 const scores=useMemo(()=>metric(rows,row=>row.hydra_mm),[rows]);
 const averageSpread=rows.length?rows.reduce((sum,row)=>sum+row.spread_mm,0)/rows.length:0;
 if(error)return <p role="alert">{error}</p>;
 if(!data)return <p className="loading">Preparing the HYDRA rainfall replay…</p>;
 if(data.status!=='available')return <div className="empty"><Database size={22}/><h3>Rainfall replay unavailable</h3><p>{data.message}</p></div>;
 return <div className="rolling-rainfall">
  <section className="replay-hero">
   <div><small>OVERVIEW · HYDRA ROLLING BACKTEST</small><h3>Six-month rainfall replay</h3><p>{data.message}</p></div>
   <label className="replay-state"><span>STATE / UT</span><select aria-label="Rainfall replay state" value={state} onChange={event=>setState(event.target.value)}>{states.map(option=><option key={option.code} value={option.name}>{option.name}</option>)}</select></label>
  </section>
  <div className="replay-facts">
   <div><CalendarDays size={15}/><span>Window</span><b>{data.coverage?.start} → {data.coverage?.end}</b></div>
   <div><Activity size={15}/><span>Issue rule</span><b>{data.history_days} prior days → +{data.lead_hours}h</b></div>
   <div><Scale size={15}/><span>HYDRA replay MAE</span><b>{fmt(scores.mae)} {data.unit}</b></div>
   <div><Database size={15}/><span>State grid</span><b>{fmt(data.grid_cell_count,0)} ERA5 cells</b></div>
  </div>
  <ChartCard title={`${data.state} · daily rainfall`} subtitle={`Actual ERA5 state mean vs HYDRA’s +${data.lead_hours}h neural-gated blend · ${data.unit}`}>
   <TimeSeries labels={rows.map(row=>row.valid_date)} unit={data.unit} band={{lower:rows.map(row=>row.interval80[0]),upper:rows.map(row=>row.interval80[1]),label:'HYDRA empirical 80% interval',color:'#d6a85f'}} series={[
    {label:'Actual ERA5 rainfall',values:rows.map(row=>row.actual_mm),color:'#dbe4de',width:2.5},
    {label:'HYDRA adaptive blend',values:rows.map(row=>row.hydra_mm),color:'#d6a85f',width:2.2},
    ...experts.map(expert=>({label:label(expert.name),values:expert.values,color:['#76a394','#8f9fba','#b199b9','#d26e55'][expert.index],dash:'5 4',width:1.1})),
   ]}/>
  </ChartCard>
  <section className="replay-method">
   <div><small>HOW TO READ THIS</small><h3>Forecast first. Actual data second.</h3><p>For every date, HYDRA issued its rainfall value using the preceding {data.history_days} days only. The actual ERA5 rainfall shown in white was held back until comparison. The four coloured lines are the real experts supplied to HYDRA’s trained neural gate; the gold line is its learned weighted blend.</p></div>
   <dl><div><dt>Training cutoff</dt><dd>{data.training_cutoff}</dd></div><div><dt>Mean expert spread</dt><dd>{fmt(averageSpread)} {data.unit}</dd></div><div><dt>Replay RMSE</dt><dd>{fmt(scores.rmse)} {data.unit}</dd></div><div><dt>Interval method</dt><dd>Pre-replay p90 residual</dd></div></dl>
  </section>
  <p className="footnote">{data.calibration} Expert spread measures disagreement between HYDRA’s inputs; it is not a confidence percentage. No provider forecast is used in this replay.</p>
 </div>;
}
