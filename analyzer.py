import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io
from datetime import datetime

# Baseline analytical cutoff. Update this to shift the entire tool's floor date.
ANALYTICAL_BASELINE = pd.Timestamp("2024-12-01")

# Keywords used to detect "new business" vs "renewal" in an explicit type column
NEW_BUSINESS_KEYWORDS = ["new", "nb", "new business", "acquisition", "acquired"]
RENEWAL_KEYWORDS = ["renewal", "renew", "renewed", "existing", "ren"]


class BookOfBusinessAnalyzer:
    def __init__(self, file_bytes: bytes, file_name: str):
        self.file_name = file_name
        
        if file_name.endswith('.csv'):
            self.df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
        elif file_name.endswith(('.xls', '.xlsx')):
            self.df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel.")
        
        self.df.columns = [str(c).strip() for c in self.df.columns]
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].astype(str).str.strip()

    def _normalize_categorical_value(self, x):
        try:
            if isinstance(x, (float, int)) and not pd.isna(x) and x == int(x):
                return str(int(x))
        except (ValueError, TypeError):
            pass
        return str(x).strip()

    _normalize_pc_value = _normalize_categorical_value

    def _classify_business_type(self, value: str) -> str:
        """Classify a business type value as 'New', 'Renewal', or 'Other'."""
        if value is None:
            return "Other"
        v = str(value).strip().lower()
        if not v or v == 'nan':
            return "Other"
        # Order matters: check renewal first so "new" doesn't false-hit "renew"
        for kw in RENEWAL_KEYWORDS:
            if kw in v:
                return "Renewal"
        for kw in NEW_BUSINESS_KEYWORDS:
            if kw in v:
                return "New"
        return "Other"

    def get_unique_column_values(self, col: str, limit: int = 500) -> list:
        if not col or col not in self.df.columns:
            return []
        raw = self.df[col].dropna()
        cleaned = [self._normalize_categorical_value(x) for x in raw]
        cleaned = [v for v in cleaned if v and v.lower() != 'nan']
        return sorted(list(set(cleaned)))[:limit]

    def get_profit_centers(self, pc_col: str) -> list:
        return self.get_unique_column_values(pc_col)

    def get_agency_codes(self, agency_col: str) -> list:
        return self.get_unique_column_values(agency_col)

    def get_date_range(self, time_col: str) -> dict:
        if not time_col or time_col not in self.df.columns:
            return {"min_date": None, "max_date": None}
        parsed = pd.to_datetime(self.df[time_col], errors='coerce').dropna()
        parsed = parsed[parsed >= ANALYTICAL_BASELINE]
        if parsed.empty:
            return {"min_date": None, "max_date": None}
        return {
            "min_date": parsed.min().strftime('%Y-%m-%d'),
            "max_date": parsed.max().strftime('%Y-%m-%d')
        }

    def infer_schema(self) -> dict:
        schema = {
            "financial_metric": None,
            "timeline_metric": None,
            "categorical_segment": None,
            "profit_center": None,
            "client_id": None,
            "agency_code": None,
            "business_type": None
        }
        
        columns = self.df.columns.tolist()
        
        for col in columns:
            col_lower = str(col).lower()
            if not schema["financial_metric"] and any(kw in col_lower for kw in ["premium", "revenue", "sales", "amount", "volume", "gwp"]):
                schema["financial_metric"] = col
                continue
            if not schema["timeline_metric"] and any(kw in col_lower for kw in ["date", "effective", "renewal date", "inception"]):
                schema["timeline_metric"] = col
                continue
            if not schema["business_type"] and any(kw in col_lower for kw in ["business type", "biz type", "new/renewal", "nb/ren", "transaction type", "policy type"]):
                schema["business_type"] = col
                continue
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

        # Fallbacks
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

        return {
            "columns": columns, 
            "inferred_mapping": schema,
            "profit_centers": self.get_profit_centers(schema["profit_center"])[:100],
            "agency_codes": self.get_agency_codes(schema["agency_code"])[:500],
            "date_range": self.get_date_range(schema["timeline_metric"]),
            "baseline_date": ANALYTICAL_BASELINE.strftime('%Y-%m-%d'),
            "has_business_type": schema["business_type"] is not None
        }

    def _compute_goal_progress(self, actual_value, goal_value, start_date, end_date, goal_period_days: int = 365) -> dict:
        if not goal_value or goal_value <= 0:
            return {
                "goal": 0, "actual": float(actual_value), "achievement_pct": 0,
                "expected_pct": 0, "gap_to_goal": 0, "projected_year_end": 0,
                "days_elapsed": 0, "status": "no_goal_set"
            }
        try:
            start = pd.to_datetime(start_date) if start_date else None
            end = pd.to_datetime(end_date) if end_date else None
        except Exception:
            start = end = None
        days_elapsed = 0
        if start and end and end >= start:
            days_elapsed = max(1, (end - start).days + 1)
        expected_pct = min(100.0, (days_elapsed / goal_period_days) * 100) if goal_period_days > 0 else 0
        achievement_pct = (actual_value / goal_value) * 100 if goal_value > 0 else 0
        gap_to_goal = float(actual_value - goal_value)
        projected_year_end = float(actual_value)
        if days_elapsed > 0 and goal_period_days > 0:
            daily_rate = actual_value / days_elapsed
            projected_year_end = float(daily_rate * goal_period_days)
        if expected_pct <= 0:
            status = "no_time_elapsed"
        else:
            pace_ratio = achievement_pct / expected_pct
            if pace_ratio >= 1.0: status = "ahead"
            elif pace_ratio >= 0.8: status = "on_pace"
            else: status = "behind"
        return {
            "goal": float(goal_value), "actual": float(actual_value),
            "achievement_pct": float(achievement_pct), "expected_pct": float(expected_pct),
            "gap_to_goal": gap_to_goal, "projected_year_end": projected_year_end,
            "days_elapsed": int(days_elapsed), "status": status
        }

    def _assign_business_type(self, working_df, id_col, time_col, biz_type_col):
        """Assign each row a 'BusinessType' column of 'New' or 'Renewal'.
        Uses explicit column if available, otherwise derives from first-seen date."""
        if biz_type_col and biz_type_col in working_df.columns:
            # Explicit method
            working_df['BusinessType'] = working_df[biz_type_col].apply(self._classify_business_type)
            # Any 'Other' values fall through to derived logic if we have id + time
            other_mask = working_df['BusinessType'] == 'Other'
            if other_mask.any() and id_col and id_col in working_df.columns:
                first_seen = self.df.copy()
                first_seen[time_col] = pd.to_datetime(first_seen[time_col], errors='coerce')
                first_seen = first_seen.dropna(subset=[time_col])
                account_birthdays = first_seen.groupby(id_col)[time_col].min()
                
                def derive(row):
                    acc = row[id_col]
                    if acc in account_birthdays.index:
                        birthday = account_birthdays[acc]
                        # If this row's date is within 30 days of first-seen, call it New
                        if pd.notna(row[time_col]) and abs((row[time_col] - birthday).days) <= 30:
                            return 'New'
                        return 'Renewal'
                    return 'Renewal'
                
                working_df.loc[other_mask, 'BusinessType'] = working_df.loc[other_mask].apply(derive, axis=1)
        elif id_col and id_col in working_df.columns:
            # Derived method: first appearance across the entire dataset = New
            first_seen = self.df.copy()
            first_seen[time_col] = pd.to_datetime(first_seen[time_col], errors='coerce')
            first_seen = first_seen.dropna(subset=[time_col])
            account_birthdays = first_seen.groupby(id_col)[time_col].min()
            
            def classify_row(row):
                acc = row[id_col]
                if acc in account_birthdays.index:
                    birthday = account_birthdays[acc]
                    if pd.notna(row[time_col]) and abs((row[time_col] - birthday).days) <= 30:
                        return 'New'
                    return 'Renewal'
                return 'Renewal'
            
            working_df['BusinessType'] = working_df.apply(classify_row, axis=1)
        else:
            # No way to classify: default everything to 'Renewal'
            working_df['BusinessType'] = 'Renewal'
        return working_df

    def run_analysis(self, mapping: dict, selected_profit_center: str = "ALL",
                     projection_target: str = "premium",
                     start_date: str = None, end_date: str = None,
                     include_future_dates: bool = False,
                     selected_agency_codes: list = None,
                     goal_value: float = 0,
                     business_view: str = "all") -> dict:
        """Executes portfolio metrics with all applied slicers.
        
        Args:
            business_view: 'all' | 'new' | 'renewal' - filters by business classification
        """
        fin_col = mapping.get("financial_metric")
        time_col = mapping.get("timeline_metric")
        cat_col = mapping.get("categorical_segment")
        pc_col = mapping.get("profit_center")
        id_col = mapping.get("client_id")
        agency_col = mapping.get("agency_code")
        biz_type_col = mapping.get("business_type")

        if not fin_col or not time_col:
            raise ValueError("Financial and Timeline metrics are required fields.")

        cols_to_keep = [fin_col, time_col]
        if cat_col: cols_to_keep.append(cat_col)
        if pc_col: cols_to_keep.append(pc_col)
        if id_col: cols_to_keep.append(id_col)
        if agency_col: cols_to_keep.append(agency_col)
        if biz_type_col: cols_to_keep.append(biz_type_col)
        cols_to_keep = list(dict.fromkeys(cols_to_keep))
        
        working_df = self.df[cols_to_keep].copy()
        working_df[time_col] = pd.to_datetime(working_df[time_col], errors='coerce')
        working_df[fin_col] = pd.to_numeric(working_df[fin_col], errors='coerce').fillna(0)
        working_df = working_df.dropna(subset=[time_col])

        if pc_col: working_df[pc_col] = working_df[pc_col].apply(self._normalize_categorical_value)
        if agency_col: working_df[agency_col] = working_df[agency_col].apply(self._normalize_categorical_value)
        if cat_col: working_df[cat_col] = working_df[cat_col].fillna('Unknown')
        if id_col: working_df[id_col] = working_df[id_col].fillna('Unknown')

        # Baseline cutoff
        working_df = working_df[working_df[time_col] >= ANALYTICAL_BASELINE]

        # Future date toggle
        future_records_removed = 0
        future_dollar_amount = 0.0
        if not include_future_dates:
            today = pd.Timestamp(datetime.now().date())
            future_mask = working_df[time_col] > today
            future_records_removed = int(future_mask.sum())
            future_dollar_amount = float(working_df.loc[future_mask, fin_col].sum())
            working_df = working_df[~future_mask]

        # Date range
        effective_start = None
        effective_end = None
        if start_date:
            try:
                start_dt = pd.to_datetime(start_date, errors='coerce')
                if pd.notna(start_dt):
                    working_df = working_df[working_df[time_col] >= start_dt]
                    effective_start = start_dt
            except Exception: pass
        if end_date:
            try:
                end_dt = pd.to_datetime(end_date, errors='coerce')
                if pd.notna(end_dt):
                    end_capped = end_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)
                    working_df = working_df[working_df[time_col] <= end_capped]
                    effective_end = end_dt
            except Exception: pass
        if effective_start is None and not working_df.empty:
            effective_start = working_df[time_col].min()
        if effective_end is None and not working_df.empty:
            effective_end = working_df[time_col].max()

        # Profit center slicer
        if pc_col and selected_profit_center and str(selected_profit_center).upper() != "ALL":
            normalized_selection = self._normalize_categorical_value(selected_profit_center)
            working_df = working_df[working_df[pc_col] == normalized_selection]

        # Agency codes slicer
        agency_codes_applied = 0
        if agency_col and selected_agency_codes and len(selected_agency_codes) > 0:
            normalized_selections = [self._normalize_categorical_value(c) for c in selected_agency_codes]
            working_df = working_df[working_df[agency_col].isin(normalized_selections)]
            agency_codes_applied = len(normalized_selections)

        # Business type classification (BEFORE view filtering so we can compute splits)
        working_df = self._assign_business_type(working_df, id_col, time_col, biz_type_col)

        # Compute the FULL New vs Renewal split before view-mode filter
        nb_df = working_df[working_df['BusinessType'] == 'New']
        ren_df = working_df[working_df['BusinessType'] == 'Renewal']
        
        business_split = {
            "new_business_premium": float(nb_df[fin_col].sum()) if not nb_df.empty else 0.0,
            "renewal_premium": float(ren_df[fin_col].sum()) if not ren_df.empty else 0.0,
            "new_business_count": int(nb_df[id_col].nunique()) if id_col and not nb_df.empty else int(len(nb_df)),
            "renewal_count": int(ren_df[id_col].nunique()) if id_col and not ren_df.empty else int(len(ren_df)),
            "classification_method": "explicit" if biz_type_col else ("derived" if id_col else "none")
        }

        # Apply view-mode filter (affects everything downstream)
        if business_view == "new":
            working_df = working_df[working_df['BusinessType'] == 'New']
        elif business_view == "renewal":
            working_df = working_df[working_df['BusinessType'] == 'Renewal']
        # "all" = no filter

        if working_df.empty:
            return {
                "kpis": {"total_premium": 0, "total_accounts": 0, "avg_account_size": 0, "retention_rate": 0, "hhi_index": 0, "pareto_ratio": 0},
                "historical_timeline": {"labels": [], "values": [], "rolling_avg": [], "mom_growth": [], "new_values": [], "renewal_values": []},
                "segment_distribution": {}, "seasonality": {}, "projections": [], "anomalies": [],
                "vintage_cohorts": {},
                "goal_progress": self._compute_goal_progress(0, goal_value, effective_start, effective_end),
                "business_split": business_split,
                "diagnostics": {
                    "future_records_removed": future_records_removed,
                    "future_dollar_amount": future_dollar_amount,
                    "include_future_dates": include_future_dates,
                    "agency_codes_applied": agency_codes_applied,
                    "business_view": business_view
                }
            }

        # 1. KPIs
        total_premium = float(working_df[fin_col].sum())
        total_accounts = int(working_df[id_col].nunique()) if id_col else int(len(working_df))
        avg_account_size = float(working_df[fin_col].mean()) if total_accounts > 0 else 0

        # 2. HHI
        hhi_index = 0
        if cat_col and total_premium > 0:
            shares = working_df.groupby(cat_col)[fin_col].sum()
            hhi_index = float(sum([(v / total_premium * 100) ** 2 for v in shares]))

        # 3. YoY Retention
        working_df['Year'] = working_df[time_col].dt.year
        retention_rate = 100.0
        years_present = sorted(working_df['Year'].unique())
        if len(years_present) >= 2 and id_col:
            prev_year_accounts = set(working_df[working_df['Year'] == years_present[-2]][id_col].unique())
            curr_year_accounts = set(working_df[working_df['Year'] == years_present[-1]][id_col].unique())
            if prev_year_accounts:
                retained = prev_year_accounts.intersection(curr_year_accounts)
                retention_rate = float(len(retained) / len(prev_year_accounts) * 100)

        # 4. Pareto
        pareto_ratio = 20.0
        if id_col and total_premium > 0:
            account_sums = working_df.groupby(id_col)[fin_col].sum().sort_values(ascending=False)
            cumulative_sum = account_sums.cumsum()
            cutoff = total_premium * 0.80
            top_accounts_count = len(cumulative_sum[cumulative_sum <= cutoff]) + 1
            pareto_ratio = float((top_accounts_count / len(account_sums)) * 100) if len(account_sums) > 0 else 20.0

        # 5. Vintage cohorts
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

        # 6. Timeline (with NB vs Renewal series overlays)
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

        # Split monthly timeline by business type for overlay
        nb_monthly = working_df[working_df['BusinessType'] == 'New'].groupby('YearMonth')
        ren_monthly = working_df[working_df['BusinessType'] == 'Renewal'].groupby('YearMonth')
        
        if projection_target == "premium":
            nb_series = nb_monthly[fin_col].sum()
            ren_series = ren_monthly[fin_col].sum()
        else:
            nb_series = nb_monthly[id_col].nunique() if id_col else nb_monthly.size()
            ren_series = ren_monthly[id_col].nunique() if id_col else ren_monthly.size()
        
        new_values = []
        renewal_values = []
        for ym in monthly_df['YearMonth']:
            new_values.append(float(nb_series.get(ym, 0)))
            renewal_values.append(float(ren_series.get(ym, 0)))

        # 7. Segment distribution
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

        # 11. Goal progress
        actual_for_goal = total_premium if projection_target == "premium" else total_accounts
        goal_progress = self._compute_goal_progress(actual_for_goal, goal_value, effective_start, effective_end)
        goal_progress["metric_type"] = projection_target

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
                "mom_growth": monthly_df['MoM_Growth'].map(float).tolist(),
                "new_values": new_values,
                "renewal_values": renewal_values
            },
            "segment_distribution": segment_data,
            "seasonality": seasonality,
            "projections": projections,
            "anomalies": anomalies,
            "vintage_cohorts": vintage_cohorts,
            "goal_progress": goal_progress,
            "business_split": business_split,
            "diagnostics": {
                "future_records_removed": future_records_removed,
                "future_dollar_amount": future_dollar_amount,
                "include_future_dates": include_future_dates,
                "agency_codes_applied": agency_codes_applied,
                "business_view": business_view
            }
        }
