import time
from pathlib import Path

import numpy as np
import pandas as pd

from harbin.advanced import AdvancedFeatureStore, _cache_is_fresh
from harbin.feature_audit import audit_feature_stack
from harbin.models import feature_columns


def _store():
    s=AdvancedFeatureStore.__new__(AdvancedFeatureStore)
    s.start_season=2026; s.end_season=2026; s.cache=Path('/tmp/no-cache')
    s.errors=[]; s.sources=[]; s.cache_diagnostics=[]; s.id_lookup={}; s.name_lookup={}
    s.static_lookup={}; s.feature_names=[]; s.dynamic_names=[]; s.identity_diagnostics={}; s.source_rows={}
    return s


def test_cache_freshness(tmp_path):
    p=tmp_path/'x.csv'; p.write_text('a,b\n1,2\n' + ('0,0\n'*40))
    assert _cache_is_fresh(p,1.0)
    old=time.time()-7200
    import os
    os.utime(p,(old,old))
    assert not _cache_is_fresh(p,1.0)
    assert _cache_is_fresh(p,None)


def test_asof_dynamic_features_exclude_current_week_and_add_recent_form():
    s=_store()
    df=pd.DataFrame({
        'season':[2026,2026,2026], 'week':[1,2,3], 'team_id':[1,1,1], 'team':['Alpha','Alpha','Alpha'],
        'epa_per_play':[0.10,0.30,0.50],
    })
    s._build_asof(df)
    w2=s.id_lookup[(2026,2,'1')]
    w3=s.id_lookup[(2026,3,'1')]
    assert abs(w2['adv_epa_per_play']-0.10)<1e-12
    assert abs(w2['recent_adv_epa_per_play']-0.10)<1e-12
    assert abs(w3['adv_epa_per_play']-0.20)<1e-12
    assert abs(w3['recent_adv_epa_per_play']-0.17)<1e-12
    assert 'recent_adv_epa_per_play' in s.dynamic_names


def test_enrich_emits_pair_coverage_features():
    s=_store(); s.dynamic_names=['adv_epa']; s.feature_names=['adv_epa']
    s.id_lookup[(2026,5,'1')]={'adv_epa':0.2}; s.id_lookup[(2026,5,'2')]={'adv_epa':-0.1}
    frame=pd.DataFrame([{'season':2026,'week':5,'home_id':'1','away_id':'2','home_team':'A','away_team':'B'}])
    out,meta=s.enrich(frame)
    assert out.loc[0,'advanced_pair_coverage']==1.0
    assert out.loc[0,'diff_adv_epa']==0.3
    assert meta['dynamic_pair_feature_coverage']==1.0


def test_feature_columns_fail_closed_on_market_and_postgame_fields():
    n=100
    df=pd.DataFrame({
        'baseline_margin':np.linspace(-5,5,n), 'elo_diff_home':np.arange(n),
        'market_spread_home':np.linspace(-3,3,n), 'closing_total':np.linspace(40,60,n),
        'target_margin_home':np.linspace(-7,7,n), 'quant_ev':np.linspace(0,.1,n),
    })
    cols=feature_columns(df)
    assert 'baseline_margin' in cols and 'elo_diff_home' in cols
    assert 'market_spread_home' not in cols
    assert 'closing_total' not in cols
    assert 'target_margin_home' not in cols
    assert 'quant_ev' not in cols


def test_feature_audit_catches_live_parity_and_leakage():
    train=pd.DataFrame({'x':[1.,2.,3.], 'market_total':[50.,51.,52.]})
    live=pd.DataFrame({'x':[4.,5.]})
    report=audit_feature_stack(train,live,['x','market_total'],{})
    assert report['status']=='FAIL'
    codes={x['code'] for x in report['issues']}
    assert 'missing_live_features' in codes
    assert 'leakage_features' in codes
