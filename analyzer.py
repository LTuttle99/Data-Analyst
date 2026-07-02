import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io
from datetime import datetime

class BookOfBusinessAnalyzer:
    def __init__(self, file_bytes: bytes, file_name: str):
        self.file_name = file_name
        
        if file_name.endswith('.csv'):
            self.df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
        elif file_name.endswith(('.xls', '.xlsx')):
            self.df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel.")
        
        # Globally normalize string formats and spaces
        self.df.columns = [str(c).strip() for c in self.df.columns]
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].astype(str).str.strip()

    def _normalize_categorical_value(self, x):
        """Consistent normalization for categorical values (profit center, agency code, etc.).
        Handles integers-stored-as-floats, whitespace, and casing consistently."""
        try:
            if isinstance(x, (float, int)) and not pd.isna(x) and x == int(x):
                return str(int(x))
        except (ValueError, TypeError):
            pass
        return str(x).strip()

    # Backwards-compatible alias so old callers still work
    _normalize_pc_value = _normalize_categorical_value

    def get_unique_column_values(self, col: str, limit: int = 500) -> list:
        """Generic helper to return unique normalized values from any column.
        Used for both profit center and agency code slicer population."""
        if not col or col not in self.df.columns:
            return []
        raw = self.df[col].dropna()
        cleaned = [self._normalize_categorical_value(x) for x in raw]
        cleaned = [v for v in cleaned if v and v.lower() != 'nan']
        return sorted(list(set(cleaned)))[:limit]

    def get_profit_centers(self, pc_col: str) -> list:
        """Backwards-compat wrapper around the generic helper."""
        return self.get_unique_column_values(pc_col)

    def get_agency_codes(self, agency_col: str) -> list:
        """Return unique normalized agency code values."""
        return self.get_unique_column_values(agency_col)

    def get_date_range(self, time_col: str) -> dict:
        """Return the min/max date bounds for a given timeline column."""
        if not time_col or time_col not in self.df.columns:
            return {"min_date": None, "max_date": None}
        
        parsed = pd.to_datetime(self.df[time_col], errors='coerce').dropna()
        parsed = parsed[parsed.dt.year >= 2022]
        
        if parsed.empty:
            return {"min_date": None, "max_date": None}
        
        return {
            "min_date": parsed.min().strftime('%Y-%m-%d'),
            "max_date": parsed.max().strftime('%Y-%m-%d')
        }

    def infer_schema(self) -> dict:
        """Heuristically identifies column roles based on names and data types."""
        schema = {
            "financial_metric": None,
            "timeline_metric": None,
            "categorical_segment": None,
            "profit_center": None,
            "client_id": None,
            "agency_code": None
        }
        
        columns = self.df.columns.tolist()
        
        for col in columns:
            col_lower = str(col).lower()
            if not schema["financial_metric"] and any(kw in col_lower for kw in ["premium", "revenue", "sales", "amount", "volume", "gwp"]):
                schema["financial_metric"] = col
                continue
            if not schema["timeline_metric"] and any(kw in col_lower for kw in ["date", "effective", "renewal", "inception", "year"]):
                schema["timeline_metric"] = col
                continue
            # Agency code inference runs BEFORE profit center to avoid "code" matching profit-center rules
            if not schema["agency_code"] and any(kw in col_lower for kw in ["agency", "agent code", "agency code", "producer code", "office code"]):
                schema["agency_code"] = col
                continue
            if not schema["profit_center"] and any(kw in col_lower for kw in ["profit", "center", "pc", "department", "unit", "branch"]):
                schema["profit_center"] = col
                continue
            if not schema["categorical_segment"] and any(kw in col_lower for kw in ["broker", "region", "product", "lob", "type", "segment", "state"]):
                schema["categorical_segment"] = col
                continue
            if not schema["client_id"] and any(kw in col_lower for kw in ["id", "client", "customer", "account", "name"]):
                schema["client_id"] = col

        # Fallbacks (agency_code intentionally has no fallback - stays None if not detected)
        if not schema["financial_metric"]:
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0: schema["financial_metric"] = num_cols[0]
        if not schema["timeline_metric"]:
            date_cols = self.df.select_dtypes(include=['datetime', 'object']).columns
            if len(date_cols) > 0: schema["timeline_metric"] = date_cols[0]
        if not schema["profit_center"]:
            cat_cols = self.df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 1: schema["profit_center"] = cat_cols[1]
        if not schema["categorical_segment"]:
            cat_cols = self.df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0: schema["categorical_segment"] = cat_cols[0]

        unique_profit_centers = self.get_profit_centers(schema["profit_center"])
        unique_agency_codes = self.get_agency_codes(schema["agency_code"])
        date_range = self.get_date_range(schema["timeline_metric"])

        return {
            "columns": columns, 
            "inferred_mapping": schema,
            "profit_centers": unique_profit_centers[:100],
            "agency_codes": unique_agency_codes[:500],
            "date_range": date_range
        }

    def run_analysis(self, mapping: dict, selected_profit_center: str = "ALL",
                     projection_target: str = "premium",
                     start_date: str = None, end_date: str = None,
                     include_future_dates: bool = False,
                     selected_agency_codes: list = None) -> dict:
        """Executes portfolio metrics with all applied slicers."""
        fin_col = mapping.get("financial_metric")
        time_col = mapping.get("timeline_metric")
        cat_col = mapping.get("categorical_segment")
        pc_col = mapping.get("profit_center")
        id_col = mapping.get("client_id")
        agency_col = mapping.get("agency_code")

        if not fin_col or not time_col:
            raise ValueError("Financial and Timeline metrics are required fields.")

        cols_to_keep = [fin_col, time_col]
        if cat_col: cols_to_keep.append(cat_col)
        if pc_col: cols_to_keep.append(pc_col)
        if id_col: cols_to_keep.append(id_col)
        if agency_col: cols_to_keep.append(agency_col)
        cols_to_keep = list(dict.fromkeys(cols_to_keep))
        
        working_df = self.df[cols_to_keep].copy()
        working_df[time_col] = pd.to_datetime(working_df[time_col], errors='coerce')
        working_df[fin_col] = pd.to_numeric(working_df[fin_col], errors='coerce').fillna(0)
        working_df = working_df.dropna(subset=[time_col])

        if pc_col:
            working_df[pc_col] = working_df[pc_col].apply(self._normalize_categorical_value)
        if agency_col:
            working_df[agency_col] = working_df[agency_col].apply(self._normalize_categorical_value)
        if cat_col: working_df[cat_col] = working_df[cat_col].fillna('Unknown')
        if id_col: working_df[id_col] = working_df[id_col].fillna('Unknown')

        # Baseline 2022 analytical cutoff
        working_df = working_df[working_df[time_col].dt.year >= 2022]

        # Future Effective Date toggle
        future_records_removed = 0
        future_dollar_amount = 0.0
        if not include_future_dates:
            today = pd.Timestamp(datetime.now().date())
            future_mask = working_df[time_col] > today
            future_records_removed = int(future_mask.sum())
            future_dollar_amount = float(working_df.loc[future_mask, fin_col].sum())
            working_df = working_df[~future_mask]

        # Date range slicer
        if start_date:
            try:
                start_dt = pd.to_datetime(start_date, errors='coerce')
                if pd.notna(start_dt):
                    working_df = working_df[working_df[time_col] >= start_dt]
            except Exception:
                pass
        if end_date:
            try:
                end_dt = pd.to_datetime(end_date, errors='coerce')
                if pd.notna(end_dt):
                    end_dt = end_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)
                    working_df = working_df[working_df[time_col] <= end_dt]
            except Exception:
                pass

        # Profit center slicer
        if pc_col and selected_profit_center and str(selected_profit_center).upper() != "ALL":
            normalized_selection = self._normalize_categorical_value(selected_profit_center)
            working_df = working_df[working_df[pc_col] == normalized_selection]

        # NEW: Agency code multi-select slicer
        # If a non-empty list is provided, filter to those codes only.
        # An empty list or None means "all agency codes."
        agency_codes_applied = 0
        if agency_col and selected_agency_codes and len(selected_agency_codes) > 0:
            normalized_selections = [
                self._normalize_categorical_value(code) for code in selected_agency_codes
            ]
            working_df = working_df[working_df[agency_col].isin(normalized_selections)]
            agency_codes_applied = len(normalized_selections)

        if working_df.empty:
            return {
                "kpis": {"total_premium": 0, "total_accounts": 0, "avg_account_size": 0, "retention_rate": 0, "hhi_index": 0, "pareto_ratio": 0},
                "historical_timeline": {"labels": [], "values": [], "rolling_avg": [], "mom_growth": []},
                "segment_distribution": {}, "seasonality": {}, "projections": [], "anomalies": [],
                "vintage_cohorts": {},
                "diagnostics": {
                    "future_records_removed": future_records_removed,
                    "future_dollar_amount": future_dollar_amount,
                    "include_future_dates": include_future_dates,
                    "agency_codes_applied": agency_codes_applied
                }
            }

        # 1. KPI Indicators
        total_premium = float(working_df[fin_col].sum())
        total_accounts = int(working_df[id_col].nunique()) if id_col else int(len(working_df))
        avg_account_size = float(working_df[fin_col].mean()) if total_accounts > 0 else 0

        # 2. HHI Concentration
        hhi_index = 0
        if cat_col and total_premium > 0:
            shares = working_df.groupby(cat_col)[fin_col].sum()
            hhi_index = float(sum([(v / total_premium * 100) ** 2 for v in shares]))

        # 3. Year-Over-Year Retention
        working_df['Year'] = working_df[time_col].dt.year
        retention_rate = 100.0
        years_present = sorted(working_df['Year'].unique())
        if len(years_present) >= 2 and id_col:
            prev_year_accounts = set(working_df[working_df['Year'] == years_present[-2]][id_col].unique())
            curr_year_accounts = set(working_df[working_df['Year'] == years_present[-1]][id_col].unique())
            if prev_year_accounts:
                retained = prev_year_accounts.intersection(curr_year_accounts)
                retention_rate = float(len(retained) / len(prev_year_accounts) * 100)

        # 4. Pareto 80/20
        pareto_ratio = 20.0
        if id_col and total_premium > 0:
            account_sums = working_df.groupby(id_col)[fin_col].sum().sort_values(ascending=False)
            cumulative_sum = account_sums.cumsum()
            cutoff = total_premium * 0.80
            top_accounts_count = len(cumulative_sum[cumulative_sum <= cutoff]) + 1
            pareto_ratio = float((top_accounts_count / len(account_sums)) * 100) if len(account_sums) > 0 else 20.0

        # 5. Vintage Cohorts
        vintage_cohorts = {}
        if id_col:
            first_seen = self.df.copy()
            first_seen[time_col] = pd.to_datetime(first_seen[time_col], errors='coerce')
            first_seen = first_seen.dropna(subset=[time_col])
            account_birthdays = first_seen.groupby(id_col)[time_col].min().dt.year.to_dict()
            
            working_df['Vintage'] = working_df[id_col].map(account_birthdays)
            vintage_summary = working_df.groupby(['Vintage', 'Year'])[fin_col if projection_target == "premium" else id_col].agg('sum' if projection_target == "premium" else 'nunique').reset_index()
            
            for v in vintage_summary['Vintage'].dropna().unique():
                v_str = f"Vintage {int(v)}"
                v_data = vintage_summary[vintage_summary['Vintage'] == v]
                vintage_cohorts[v_str] = {f"CY_{int(row['Year'])}": float(row[fin_col if projection_target == "premium" else id_col]) for _, row in v_data.iterrows()}

        # 6. Timeline
        working_df['YearMonth'] = working_df[time_col].dt.to_period('M')
        monthly_groups = working_df.groupby('YearMonth')
        
        monthly_df = pd.DataFrame({
            'premium': monthly_groups[fin_col].sum(),
            'count': monthly_groups[id_col].nunique() if id_col else monthly_groups.size()
        }).reset_index()
        monthly_df['YearMonthStr'] = monthly_df['YearMonth'].astype(str)
        
        target_series = 'premium' if projection_target == "premium" else 'count'
        monthly_df['RollingAvg'] = monthly_df[target_series].rolling(window=3, min_periods=1).mean()
        monthly_df['MoM_Growth'] = monthly_df[target_series].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0) * 100

        # 7. Segment Distribution
        segment_data = {}
        if cat_col:
            seg_summary = working_df.groupby(cat_col)[fin_col if projection_target == "premium" else (id_col if id_col else fin_col)].agg('sum' if projection_target == "premium" else 'nunique').sort_values(ascending=False).head(20)
            segment_data = {str(k): float(v) for k, v in seg_summary.items()}

        # 8. Seasonality
        working_df['MonthName'] = working_df[time_col].dt.strftime('%B')
        season_summary = working_df.groupby('MonthName')[fin_col if projection_target == "premium" else (id_col if id_col else fin_col)].agg('sum' if projection_target == "premium" else 'nunique')
        seasonality = {k: float(v) for k, v in season_summary.to_dict().items()}

        # 9. Projections
        projections = []
        if len(monthly_df) > 1:
            X = np.arange(len(monthly_df)).reshape(-1, 1)
            y = monthly_df[target_series].values
            model = LinearRegression().fit(X, y)
            
            future_X = np.arange(len(monthly_df), len(monthly_df) + 12).reshape(-1, 1)
            future_predictions = model.predict(future_X)
            
            last_date = working_df[time_col].max()
            for i, pred in enumerate(future_predictions):
                next_month = (last_date + pd.DateOffset(months=i+1)).strftime('%Y-%m')
                projections.append({"period": next_month, "projected_value": max(0.0, float(pred))})

        # 10. Anomalies
        anomalies = []
        if id_col:
            top_accounts = working_df.groupby(id_col)[fin_col].sum().sort_values(ascending=False).head(10)
            for acc_id, acc_vol in top_accounts.items():
                if total_premium > 0 and (acc_vol / total_premium) > 0.03:
                    anomalies.append({
                        "identifier": str(acc_id),
                        "value": float(acc_vol),
                        "reason": f"High Concentration Exposure Outlier Risk ({round(acc_vol/total_premium*100, 1)}% of total selected folder scope)"
                    })

        return {
            "kpis": {
                "total_premium": total_premium,
                "total_accounts": total_accounts,
                "avg_account_size": avg_account_size,
                "retention_rate": retention_rate,
                "hhi_index": hhi_index,
                "pareto_ratio": pareto_ratio
            },
            "historical_timeline": {
                "labels": monthly_df['YearMonthStr'].tolist(),
                "values": monthly_df[target_series].map(float).tolist(),
                "rolling_avg": monthly_df['RollingAvg'].map(float).tolist(),
                "mom_growth": monthly_df['MoM_Growth'].map(float).tolist()
            },
            "segment_distribution": segment_data,
            "seasonality": seasonality,
            "projections": projections,
            "anomalies": anomalies,
            "vintage_cohorts": vintage_cohorts,
            "diagnostics": {
                "future_records_removed": future_records_removed,
                "future_dollar_amount": future_dollar_amount,
                "include_future_dates": include_future_dates,
                "agency_codes_applied": agency_codes_applied
            }
        }
