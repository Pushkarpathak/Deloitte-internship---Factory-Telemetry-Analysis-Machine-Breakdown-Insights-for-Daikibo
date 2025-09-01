"""
Daikibo Telemetry Analysis
Single-file script to:
 - load unified telemetry JSON
 - identify breakdown events (configurable)
 - aggregate breakdown counts by factory and by machine type
 - save summary CSVs and plots

Usage example:
 python daikibo_telemetry_analysis.py --input data/telemetry_data.json --output outputs/

The script is flexible about how "breakdown" is detected: by default it tries common fields
like 'status', 'error', 'state', 'fault' and treats values containing keywords like
'error', 'failed', 'broken', 'fault' as breakdowns. You can override with --breakdown-field
and --breakdown-values if your JSON uses a custom flag.

Requirements:
 pandas
 matplotlib
"""

import argparse
import json
import os
from collections import Counter
from typing import List, Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Helper functions
# -------------------------

def load_json_to_dataframe(json_path: str) -> pd.DataFrame:
    """Loads a JSON file that contains telemetry records and returns a DataFrame.

    The function expects the JSON to be either a list of records or an object
    with a top-level key containing a list. It flattens nested dicts where possible.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # If top-level is a dict that wraps the records, try to locate the list
    if isinstance(raw, dict):
        # If raw has exactly one list-valued key, use that
        list_candidates = [v for v in raw.values() if isinstance(v, list)]
        if len(list_candidates) == 1:
            records = list_candidates[0]
        else:
            # try common keys
            for k in ("data", "records", "telemetry", "messages"):
                if k in raw and isinstance(raw[k], list):
                    records = raw[k]
                    break
            else:
                # fallback: attempt to convert dict-of-lists into rows
                try:
                    df = pd.json_normalize(raw)
                    return df
                except Exception:
                    raise ValueError("Couldn't interpret JSON structure. Provide a list of records or wrap records under a single key.")
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError("Unsupported JSON top-level structure")

    # Normalize into a flat table
    df = pd.json_normalize(records)
    return df


def infer_breakdown(df: pd.DataFrame, breakdown_field: Optional[str] = None,
                    breakdown_values: Optional[List[str]] = None) -> pd.Series:
    """Returns a boolean Series where True indicates a breakdown event.

    If breakdown_field is provided and exists in df, it will be used. Otherwise
    the function will search for likely fields and try to infer.

    breakdown_values is a list of keywords; if any keyword is found (case-insensitive)
    within the field's string value, it's considered a breakdown.
    """
    # Candidate fields to check if user didn't provide one
    candidates = [breakdown_field] if breakdown_field else []
    candidates += [c for c in ['status', 'state', 'error', 'fault', 'alarm', 'flag'] if c in df.columns]

    if not candidates:
        # fallback: check for boolean/error code columns
        for col in df.columns:
            if df[col].dropna().isin([0, 1, True, False]).any():
                candidates.append(col)

    if not candidates:
        # last resort: check all string columns
        candidates = [col for col in df.columns if df[col].dtype == object]

    # default keywords
    keywords = [k.lower() for k in (breakdown_values or ['error', 'failed', 'broken', 'fault', 'down', 'alarm', '1'])]

    # Evaluate each candidate and pick the one that yields the most matches (heuristic)
    best_series = pd.Series(False, index=df.index)
    best_count = -1
    best_col = None
    for col in candidates:
        try:
            col_series = df[col].astype(str).fillna("")
        except Exception:
            continue
        # count matches
        matches = col_series.str.lower().apply(lambda s: any(kw in s for kw in keywords))
        count = int(matches.sum())
        if count > best_count:
            best_count = count
            best_series = matches
            best_col = col

    if best_col is None:
        # nothing matched, return all False
        return pd.Series(False, index=df.index)

    # If a numeric flag column (0/1), allow that interpretation as well
    # e.g., if best_col had only 0/1 values
    try:
        unique_vals = df[best_col].dropna().unique()
        if set(unique_vals).issubset({0, 1, '0', '1', True, False}):
            # treat truthy values as breakdown
            numeric_matches = df[best_col].apply(lambda v: str(v).strip() in {'1', 'True', 'true', 'TRUE'})
            # if numeric_matches produces more positives, prefer it
            if numeric_matches.sum() >= best_series.sum():
                best_series = numeric_matches
    except Exception:
        pass

    return best_series


def aggregate_breakdowns(df: pd.DataFrame, breakdown_mask: pd.Series,
                         factory_col_candidates: List[str] = None,
                         machine_col_candidates: List[str] = None) -> Dict[str, Any]:
    """Aggregate breakdown counts by factory and machine type.

    The function attempts to identify the factory and machine-type columns by
    examining common names (e.g., 'factory', 'location', 'site') and
    ('machine_type', 'machine', 'type', 'model'). If not found, it will raise.
    """
    # Detect factory column
    if factory_col_candidates is None:
        factory_col_candidates = [c for c in ['factory', 'location', 'site', 'plant'] if c in df.columns]
    if not factory_col_candidates:
        # be permissive: look for any column with 'factory' or 'location' substring
        factory_col_candidates = [c for c in df.columns if 'factory' in c.lower() or 'location' in c.lower() or 'site' in c.lower()]

    if not factory_col_candidates:
        raise ValueError("Could not find a factory/location column in the data. Columns present: {}".format(list(df.columns)))

    factory_col = factory_col_candidates[0]

    # Detect machine column
    if machine_col_candidates is None:
        machine_col_candidates = [c for c in ['machine_type', 'machine', 'type', 'model', 'machinetype'] if c in df.columns]
    if not machine_col_candidates:
        machine_col_candidates = [c for c in df.columns if 'machine' in c.lower() or 'type' in c.lower() or 'model' in c.lower()]

    if not machine_col_candidates:
        raise ValueError("Could not find a machine-type column in the data. Columns present: {}".format(list(df.columns)))

    machine_col = machine_col_candidates[0]

    # Build a DataFrame of breakdown events
    breakdown_df = df[breakdown_mask].copy()
    if breakdown_df.empty:
        print("No breakdowns detected with the selected heuristics.")
        # still return zeros
        return {
            'by_factory': Counter(),
            'by_machine_in_worst_factory': Counter(),
            'factory_col': factory_col,
            'machine_col': machine_col,
        }

    by_factory = Counter(breakdown_df[factory_col].fillna('Unknown').astype(str).str.strip())
    # identify worst factory
    if len(by_factory) == 0:
        worst_factory = None
    else:
        worst_factory = by_factory.most_common(1)[0][0]

    # breakdowns by machine within the worst factory
    by_machine = Counter()
    if worst_factory is not None:
        mask_worst = breakdown_df[factory_col].astype(str).str.strip() == str(worst_factory)
        by_machine = Counter(breakdown_df[mask_worst][machine_col].fillna('Unknown').astype(str).str.strip())

    return {
        'by_factory': by_factory,
        'worst_factory': worst_factory,
        'by_machine_in_worst_factory': by_machine,
        'factory_col': factory_col,
        'machine_col': machine_col,
    }


def plot_counter(counter: Counter, title: str, output_path: str, top_n: Optional[int] = None):
    """Plots a bar chart of the counter and saves it to output_path."""
    if not counter:
        print(f"No data to plot for {title}")
        return

    items = counter.most_common(top_n) if top_n else counter.most_common()
    labels, values = zip(*items)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel('Breakdown Count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")


# -------------------------
# Main workflow
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze Daikibo telemetry JSON for machine breakdowns")
    parser.add_argument('--input', '-i', required=True, help='Path to telemetry JSON file')
    parser.add_argument('--output', '-o', default='outputs', help='Directory to save outputs')
    parser.add_argument('--breakdown-field', help='Field name indicating status/error (optional)')
    parser.add_argument('--breakdown-values', nargs='+', help='Keywords indicating breakdown (e.g. error failed broken)')
    parser.add_argument('--top-n', type=int, default=10, help='Top N machines to plot for worst factory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading JSON...")
    df = load_json_to_dataframe(args.input)
    print(f"Loaded {len(df)} records; columns: {list(df.columns)}")

    print("Detecting breakdown events...")
    breakdown_mask = infer_breakdown(df, breakdown_field=args.breakdown_field, breakdown_values=args.breakdown_values)
    total_breakdowns = int(breakdown_mask.sum())
    print(f"Detected {total_breakdowns} breakdown events (heuristic)")

    print("Aggregating breakdowns by factory and machine...")
    agg = aggregate_breakdowns(df, breakdown_mask)

    # Save summaries
    by_factory = agg['by_factory']
    factory_csv = os.path.join(args.output, 'breakdowns_by_factory.csv')
    pd.DataFrame(by_factory.most_common(), columns=['factory', 'breakdown_count']).to_csv(factory_csv, index=False)
    print(f"Saved: {factory_csv}")

    if agg['worst_factory'] is not None:
        worst = agg['worst_factory']
        by_machine = agg['by_machine_in_worst_factory']
        machine_csv = os.path.join(args.output, f'breakdowns_by_machine_in_{sanitize_filename(worst)}.csv')
        pd.DataFrame(by_machine.most_common(), columns=['machine_type', 'breakdown_count']).to_csv(machine_csv, index=False)
        print(f"Saved: {machine_csv}")

    # Plots
    plot_counter(by_factory, 'Breakdowns by Factory', os.path.join(args.output, 'breakdowns_by_factory.png'))
    if agg['worst_factory'] is not None:
        plot_counter(agg['by_machine_in_worst_factory'], f"Top {args.top_n} Machines in {agg['worst_factory']}",
                     os.path.join(args.output, f"top_machines_in_{sanitize_filename(agg['worst_factory'])}.png"), top_n=args.top_n)

    # Print summary to console
    print('\nSummary:')
    if by_factory:
        for fac, cnt in by_factory.most_common():
            print(f"  {fac}: {cnt}")
    else:
        print("  No breakdowns by factory detected.")

    if agg['worst_factory']:
        print(f"\nWorst factory: {agg['worst_factory']}")
        print("Top machines in worst factory:")
        for m, c in agg['by_machine_in_worst_factory'].most_common(args.top_n):
            print(f"  {m}: {c}")
    else:
        print("No single worst factory identified.")


def sanitize_filename(s: str) -> str:
    return ''.join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in str(s)).replace(' ', '_')


if __name__ == '__main__':
    main()
