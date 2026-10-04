// Dependency-free SVG charts for HYDRA. Colours match the existing dashboard palette.
import {fmt} from './api';

export const PALETTE=['#d6a85f','#76a394','#8f9fba','#b199b9','#d26e55'];
const AXIS='#81908e',GRID='#2a3435',TEXT='#c6d0cc';
const mono="'IBM Plex Mono',monospace";

export function ChartCard({title,subtitle,children}:{title:string;subtitle?:string;children:React.ReactNode}){
 return <section className="card chart-card"><h3>{title}</h3>{subtitle&&<p className="chart-sub">{subtitle}</p>}{children}</section>;
}

const tickFmt=(step:number)=>(v:number)=>fmt(v,step<1?2:Number.isInteger(step)?0:1);
function niceMax(v:number){if(!isFinite(v)||v<=0)return 1;const p=Math.pow(10,Math.floor(Math.log10(v))),n=v/p;return (n<=1?1:n<=2?2:n<=5?5:10)*p;}

/* ---------- Donut: share of a whole (e.g. expert weights) ---------- */
export type Slice={label:string;value:number;color?:string};
export function Donut({items,centerLabel='100%'}:{items:Slice[];centerLabel?:string}){
 const data=items.filter(i=>i.value>0),total=data.reduce((s,i)=>s+i.value,0);
 if(!total)return <p className="muted">No data to chart.</p>;
 const R=52,r=33,cx=60,cy=60;let a=-Math.PI/2;
 const arcs=data.map((d,i)=>{
  const sweep=Math.min(d.value/total,0.9999)*Math.PI*2,a0=a,a1=a+sweep;a=a1;
  const p=(ang:number,rad:number)=>[cx+rad*Math.cos(ang),cy+rad*Math.sin(ang)];
  const [x0,y0]=p(a0,R),[x1,y1]=p(a1,R),[x2,y2]=p(a1,r),[x3,y3]=p(a0,r),large=sweep>Math.PI?1:0;
  return {d:`M${x0},${y0} A${R},${R} 0 ${large} 1 ${x1},${y1} L${x2},${y2} A${r},${r} 0 ${large} 0 ${x3},${y3} Z`,color:d.color||PALETTE[i%PALETTE.length],item:d};
 });
 return <div className="donut-wrap"><svg viewBox="0 0 120 120" role="img" aria-label="Share chart" className="donut">
  {arcs.map(a=><path key={a.item.label} d={a.d} fill={a.color}><title>{`${a.item.label}: ${fmt(a.item.value/total*100,1)}%`}</title></path>)}
  <text x={cx} y={cy+3} textAnchor="middle" fill={TEXT} fontSize="9" fontFamily={mono}>{centerLabel}</text></svg>
  <ul className="legend">{arcs.map(a=><li key={a.item.label}><i style={{background:a.color}}/>{a.item.label}<b>{fmt(a.item.value/total*100,1)}%</b></li>)}</ul></div>;
}

/* ---------- Horizontal bars with optional reference line + band ---------- */
export type Bar={label:string;value:number|null|undefined;color?:string};
export function HBars({bars,unit,reference,referenceLabel,band,bandLabel}:{bars:Bar[];unit?:string;reference?:number|null;referenceLabel?:string;band?:[number,number]|null;bandLabel?:string}){
 const rows=bars.filter(b=>b.value!=null) as {label:string;value:number;color?:string}[];
 if(!rows.length)return <p className="muted">No data to chart.</p>;
 const vals=[...rows.map(r=>r.value),...(reference!=null?[reference]:[]),...(band?band:[])];
 const lo=Math.min(0,...vals),hi=Math.max(0,...vals),span=(hi-lo)||1,max=niceMax(Math.max(Math.abs(lo),Math.abs(hi)));
 const dmin=lo<0?-max:0,dmax=hi>0?max:0,rng=(dmax-dmin)||1;
 const L=118,Rm=48,W=520,rowH=26,T=18,H=T+rows.length*rowH+24,iw=W-L-Rm;
 const x=(v:number)=>L+((v-dmin)/rng)*iw,zero=x(0);
 const ticks=[0,.25,.5,.75,1].map(t=>dmin+t*rng),tf=tickFmt(rng/4);
 return <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Bar chart">
  {ticks.map((t,i)=><g key={i}><line x1={x(t)} x2={x(t)} y1={T-4} y2={H-22} stroke={GRID} strokeWidth="1"/><text x={x(t)} y={H-9} textAnchor="middle" fill={AXIS} fontSize="9" fontFamily={mono}>{tf(t)}</text></g>)}
  {band&&<g><rect x={x(Math.min(...band))} y={T-4} width={Math.max(2,Math.abs(x(band[1])-x(band[0])))} height={H-22-T+4} fill="#76a394" opacity=".16"><title>{bandLabel||'80% interval'}</title></rect></g>}
  {rows.map((r,i)=>{const y=T+i*rowH,bx=Math.min(zero,x(r.value)),bw=Math.max(1,Math.abs(x(r.value)-zero));
   return <g key={r.label}><text x={L-8} y={y+13} textAnchor="end" fill={TEXT} fontSize="10" fontFamily={mono}>{r.label.length>17?r.label.slice(0,16)+'…':r.label}</text>
    <rect x={bx} y={y+3} width={bw} height={15} rx="2" fill={r.color||PALETTE[i%PALETTE.length]}><title>{`${r.label}: ${fmt(r.value)} ${unit||''}`}</title></rect>
    <text x={Math.max(x(r.value),zero)+5} y={y+14} fill={TEXT} fontSize="9" fontFamily={mono}>{fmt(r.value)}</text></g>;})}
  {reference!=null&&<g><line x1={x(reference)} x2={x(reference)} y1={T-6} y2={H-22} stroke="#e9c979" strokeWidth="1.5" strokeDasharray="4 3"/><text x={x(reference)} y={T-8} textAnchor="middle" fill="#e9c979" fontSize="9" fontFamily={mono}>{referenceLabel||'blend'} {fmt(reference)}</text></g>}
  {unit&&<text x={W-4} y={H-9} textAnchor="end" fill={AXIS} fontSize="9" fontFamily={mono}>{unit}</text>}
 </svg>;
}

/* ---------- Grouped vertical bars (e.g. HYDRA vs provider, RMSE vs MAE) ---------- */
export type Group={name:string;values:{label:string;value:number|null|undefined;color?:string}[]};
export function GroupedBars({groups,unit,seriesColors}:{groups:Group[];unit?:string;seriesColors?:Record<string,string>}){
 const all=groups.flatMap(g=>g.values.map(v=>v.value).filter((v):v is number=>v!=null));
 if(!all.length)return <p className="muted">No data to chart.</p>;
 const lo=Math.min(0,...all),hi=Math.max(0,...all),max=niceMax(Math.max(Math.abs(lo),Math.abs(hi)));
 const dmin=lo<0?-max:0,dmax=hi>0?max:0,rng=(dmax-dmin)||1;
 const W=520,H=230,L=40,R=10,T=14,B=48,iw=W-L-R,ih=H-T-B,y=(v:number)=>T+(1-(v-dmin)/rng)*ih,zero=y(0);
 const gw=iw/groups.length,series=[...new Set(groups.flatMap(g=>g.values.map(v=>v.label)))];
 const colour=(s:string,i:number)=>seriesColors?.[s]||PALETTE[i%PALETTE.length];
 const ticks=[0,.25,.5,.75,1].map(t=>dmin+t*rng),tf=tickFmt(rng/4);
 return <><svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Grouped bar chart">
  {ticks.map((t,i)=><g key={i}><line x1={L} x2={W-R} y1={y(t)} y2={y(t)} stroke={GRID}/><text x={L-6} y={y(t)+3} textAnchor="end" fill={AXIS} fontSize="9" fontFamily={mono}>{tf(t)}</text></g>)}
  {groups.map((g,gi)=>{const bw=Math.min(34,(gw*.7)/Math.max(1,g.values.length)),start=L+gi*gw+(gw-bw*g.values.length)/2;
   return <g key={g.name}>{g.values.map((v,vi)=>{if(v.value==null)return null;const bx=start+vi*bw,by=Math.min(y(v.value),zero),bh=Math.max(1,Math.abs(y(v.value)-zero));
    return <g key={v.label}><rect x={bx+1} y={by} width={bw-2} height={bh} rx="2" fill={v.color||colour(v.label,series.indexOf(v.label))}><title>{`${g.name} · ${v.label}: ${fmt(v.value)} ${unit||''}`}</title></rect>
     <text x={bx+bw/2} y={v.value>=0?by-3:by+bh+10} textAnchor="middle" fill={TEXT} fontSize="8" fontFamily={mono}>{fmt(v.value)}</text></g>;})}
    <text x={L+gi*gw+gw/2} y={H-B+16} textAnchor="middle" fill={TEXT} fontSize="9" fontFamily={mono}>{g.name.length>16?g.name.slice(0,15)+'…':g.name}</text></g>;})}
  <line x1={L} x2={W-R} y1={zero} y2={zero} stroke={AXIS}/>
  {unit&&<text x={L} y={T-3} fill={AXIS} fontSize="9" fontFamily={mono}>{unit}</text>}
 </svg>
 <ul className="legend inline">{series.map((s,i)=><li key={s}><i style={{background:colour(s,i)}}/>{s}</li>)}</ul></>;
}

/* ---------- Historical timeline with an optional empirical interval ---------- */
export type LineSeries={label:string;values:(number|null|undefined)[];color?:string;dash?:string;width?:number};
export function TimeSeries({labels,series,unit,band}:{labels:string[];series:LineSeries[];unit?:string;band?:{lower:(number|null|undefined)[];upper:(number|null|undefined)[];label?:string;color?:string}}){
 const values=[...series.flatMap(s=>s.values),...(band?[...band.lower,...band.upper]:[])].filter((v):v is number=>typeof v==='number'&&isFinite(v));
 if(!values.length)return <p className="muted">No timeline data to chart.</p>;
 const W=760,H=310,L=48,R=14,T=20,B=48,iw=W-L-R,ih=H-T-B,max=niceMax(Math.max(0,...values));
 const x=(i:number)=>L+(labels.length<2?iw/2:i/(labels.length-1)*iw),y=(value:number)=>T+(1-Math.max(0,value)/max)*ih;
 const ticks=[0,.25,.5,.75,1].map(t=>max*t),tf=tickFmt(max/4);
 const path=(points:(number|null|undefined)[])=>{let open=false;return points.map((value,index)=>{if(value==null||!isFinite(value)){open=false;return ''}const command=open?'L':'M';open=true;return `${command}${x(index).toFixed(2)},${y(value).toFixed(2)}`}).join(' ')};
 const dateLabels=labels.map((value,index)=>({value,index})).filter(({value,index})=>index===0||index===labels.length-1||value.slice(8,10)==='01');
 const bandPath=band?`${path(band.upper)} ${[...band.lower].map((value,reverseIndex)=>{const index=band.lower.length-1-reverseIndex;if(value==null||!isFinite(value))return '';return `L${x(index).toFixed(2)},${y(value).toFixed(2)}`}).join(' ')} Z`:'';
 return <><svg viewBox={`0 0 ${W} ${H}`} className="chart timeline-chart" role="img" aria-label="HYDRA rainfall replay timeline">
  {ticks.map((tick,index)=><g key={index}><line x1={L} x2={W-R} y1={y(tick)} y2={y(tick)} stroke={GRID}/><text x={L-7} y={y(tick)+3} textAnchor="end" fill={AXIS} fontSize="9" fontFamily={mono}>{tf(tick)}</text></g>)}
  {band&&<path d={bandPath} fill={band.color||'#d6a85f'} opacity=".14"><title>{band.label||'HYDRA empirical 80% interval'}</title></path>}
  {series.map((line,index)=><path key={line.label} d={path(line.values)} fill="none" stroke={line.color||PALETTE[index%PALETTE.length]} strokeWidth={line.width||1.7} strokeDasharray={line.dash} strokeLinecap="round" strokeLinejoin="round"><title>{line.label}</title></path>)}
  {dateLabels.map(({value,index})=><text key={`${value}-${index}`} x={x(index)} y={H-19} textAnchor={index===0?'start':index===labels.length-1?'end':'middle'} fill={AXIS} fontSize="9" fontFamily={mono}>{new Date(`${value}T00:00:00Z`).toLocaleDateString('en-IN',{month:'short',year:'2-digit',timeZone:'UTC'})}</text>)}
  {unit&&<text x={L} y={T-6} fill={AXIS} fontSize="9" fontFamily={mono}>{unit}</text>}
 </svg><ul className="legend inline timeline-legend">{[...(band?[{label:band.label||'Empirical 80% interval',color:band.color||'#d6a85f'}]:[]),...series].map((line,index)=><li key={line.label}><i style={{background:line.color||PALETTE[index%PALETTE.length]}}/>{line.label}</li>)}</ul></>;
}
