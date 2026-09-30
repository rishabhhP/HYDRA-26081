"""Run unchanged upstream inference offline and publish an atomic forecast cycle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import data as D


def validate_history(history, grid, issue):
    import numpy as np
    import pandas as pd
    from nwpblend.features import CUBE_VARS
    from nwpblend import config as C
    needed = {'date','latitude','longitude', *CUBE_VARS, *C.FLAGS}
    missing = needed-set(history.columns)
    if missing:
        raise ValueError('History missing required columns: '+', '.join(sorted(missing)))
    h = history.copy()
    h['date'] = pd.to_datetime(h['date'])
    issue = pd.Timestamp(issue).normalize()
    h = h[(h.date<=issue)&(h.date>issue-pd.Timedelta(days=35))]
    if sorted(h.date.unique()) != list(pd.date_range(issue-pd.Timedelta(days=34),issue)):
        raise ValueError('Need 35 consecutive UTC days ending on issue day')
    if h.duplicated(['date','latitude','longitude']).any():
        raise ValueError('Duplicate grid cell/day')
    expected = grid[['latitude','longitude']].sort_values(['latitude','longitude']).to_numpy()
    for _, day in h.groupby('date'):
        actual = day[['latitude','longitude']].sort_values(['latitude','longitude']).to_numpy()
        if actual.shape != expected.shape or not np.allclose(actual,expected,rtol=0,atol=1e-6):
            raise ValueError('History must match the trained full grid on every day')
    if not np.isfinite(h[list(CUBE_VARS)+C.FLAGS].to_numpy(dtype=float)).all():
        raise ValueError('History contains missing or non-finite required atmospheric values')
    return h


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--history',type=Path,required=True)
    parser.add_argument('--issue',required=True)
    args=parser.parse_args()
    if not (D.ART/'climatology_by_doy.npz').exists():
        parser.error('Missing artifacts/blocked/climatology_by_doy.npz; use the original training data to regenerate it.')
    sys.path.insert(0,str(D.REPO))
    import pandas as pd
    from .experts import NwpblendExpert
    grid=pd.read_parquet(D.ART/'grid_static.parquet')
    history=validate_history(pd.read_parquet(args.history),grid,args.issue)
    model=NwpblendExpert(D.ART)
    output=model.infer(history,args.issue,leads=D.MANIFEST['leads_days'])
    D.RUNTIME.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    path=D.RUNTIME/f'forecast-{stamp}.csv'
    output.to_csv(path,index=False)
    def sha(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
        return h.hexdigest()
    meta={'method':'BlendingForecaster.forecast, original preprocessing','history_sha256':sha(args.history),
          'issue_date':args.issue,'generated_at':datetime.now(timezone.utc).isoformat(),
          'checkpoint_hashes':{p.relative_to(D.ART).as_posix():sha(p) for p in D.ART.rglob('*') if p.is_file()},
          'code_version':D.provenance(D.ARCHIVE)['commit']}
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    temporary=D.RUNTIME/'latest.tmp'
    temporary.write_text(json.dumps({'file':path.name}))
    os.replace(temporary,D.RUNTIME/'latest.json')
    print(f'Published {len(output)} forecast rows to {path.name}')


if __name__=='__main__':
    main()
