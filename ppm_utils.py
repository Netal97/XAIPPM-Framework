import pandas as pd
import numpy as np

def build_prefix_log(events, case_info):
    df = events.merge(case_info[['IncidentID','case_start','case_end','late']], on='IncidentID', how='inner')
    df = df.sort_values(['IncidentID','time:timestamp']).reset_index(drop=True)
    g = df.groupby('IncidentID', sort=False)

    df['prefix_len'] = g.cumcount() + 1
    df['case_len'] = g['time:timestamp'].transform('size')
    df['pct_complete'] = df['prefix_len'] / df['case_len']
    df['elapsed_h'] = (df['time:timestamp'] - df['case_start']).dt.total_seconds() / 3600
    df['remaining_h'] = ((df['case_end'] - df['time:timestamp']).dt.total_seconds() / 3600).clip(lower=0)
    df['time_since_prev_h'] = g['time:timestamp'].diff().dt.total_seconds().fillna(0) / 3600

    df['hour'] = df['time:timestamp'].dt.hour
    df['dow']  = df['time:timestamp'].dt.dayofweek
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24); df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['dow_sin']  = np.sin(2*np.pi*df['dow']/7);   df['dow_cos']  = np.cos(2*np.pi*df['dow']/7)

    dummies = pd.get_dummies(df['concept:name'], prefix='act')
    dummies_cum = dummies.groupby(df['IncidentID'], sort=False).cumsum()
    df['n_distinct_acts_so_far'] = (dummies_cum > 0).sum(axis=1)
    df['heuristic_remaining_h'] = (df['elapsed_h'] * (1/df['pct_complete'].clip(lower=0.01) - 1)).clip(lower=0)

    # --- features de rythme / dynamique du cas ---
    df['gap_mean_so_far'] = g['time_since_prev_h'].apply(lambda s: s.expanding().mean()).values
    df['gap_std_so_far']  = g['time_since_prev_h'].apply(lambda s: s.expanding().std()).fillna(0).values
    df['gap_max_so_far']  = g['time_since_prev_h'].apply(lambda s: s.expanding().max()).values
    df['gap_min_so_far']  = g['time_since_prev_h'].apply(lambda s: s.expanding().min()).values
    df['pace_ratio'] = df['time_since_prev_h'] / (df['gap_mean_so_far'] + 0.01)
    df['events_per_hour'] = df['prefix_len'] / (df['elapsed_h'] + 0.1)
    df['activity_diversity'] = df['n_distinct_acts_so_far'] / df['prefix_len']
    df['gap_cv'] = df['gap_std_so_far'] / (df['gap_mean_so_far'] + 0.01)
    df['gap_prev2'] = g['time_since_prev_h'].shift(1).fillna(0).values
    df['gap_trend'] = df['time_since_prev_h'] - df['gap_prev2']

    return df