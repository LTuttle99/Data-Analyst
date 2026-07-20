import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io
from datetime import datetime


ANALYTICAL_BASELINE = pd.Timestamp("2020-01-01")

NEW_BUSINESS_KEYWORDS = ["new", "nb", "new business", "acquisition", "acquired"]
RENEWAL_KEYWORDS = ["renewal", "renew", "renewed", "existing", "ren"]


class BookOfBusinessAnalyzer:
    """
    Loads a single uploaded book-of-business ledger (CSV/XLSX) and provides schema inference,
    filtering, KPI computation, seasonality-aware forecasting, and multi-goal pacing analysis.

    One instance corresponds to one uploaded file for the lifetime of a session; `self.df`
    holds the cleaned raw data and all analysis methods operate on filtered copies of it.
    """

    def __init__(self, file_bytes: bytes, file_name: str):
        self.file_name = file_name

        if file_name.lower().endswith(".csv"):
            self.df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
        elif file_name.lower().endswith((".xls", ".xlsx")):
            self.df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel.")

        self.df.columns = [str(c).strip() for c in self.df.columns]

        for col in self.df.select_dtypes(include=["object"]).columns:
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
        if value is None:
            return "Other"

        v = str(value).strip().lower()

        if not v or v == "nan":
            return "Other"

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
        cleaned = [v for v in cleaned if v and v.lower() != "nan"]

        return sorted(list(set(cleaned)))[:limit]

    def get_profit_centers(self, pc_col: str) -> list:
        return self.get_unique_column_values(pc_col)

    def get_agency_codes(self, agency_col: str) -> list:
        return self.get_unique_column_values(agency_col)

    def get_date_range(self, time_col: str) -> dict:
        if not time_col or time_col not in self.df.columns:
            return {"min_date": None, "max_date": None}

        parsed = pd.to_datetime(self.df[time_col], errors="coerce").dropna()
        parsed = parsed[parsed >= ANALYTICAL_BASELINE]

        if parsed.empty:
            return {"min_date": None, "max_date": None}

        return {
            "min_date": parsed.min().strftime("%Y-%m-%d"),
            "max_date": parsed.max().strftime("%Y-%m-%d")
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

            if not schema["financial_metric"] and any(
                kw in col_lower for kw in ["premium", "revenue", "sales", "amount", "volume", "gwp"]
            ):
                schema["financial_metric"] = col
                continue

            if not schema["timeline_metric"] and any(
                kw in col_lower for kw in ["date", "effective", "renewal date", "inception"]
            ):
                schema["timeline_metric"] = col
                continue

            if not schema["business_type"] and any(
                kw in col_lower for kw in [
                    "business type", "biz type", "new/renewal", "nb/ren",
                    "transaction type", "policy type"
                ]
            ):
                schema["business_type"] = col
                continue

            if not schema["agency_code"] and any(
                kw in col_lower for kw in [
                    "agency", "agent code", "agency code", "producer code", "office code"
                ]
            ):
                schema["agency_code"] = col
                continue

            if not schema["profit_center"] and any(
                kw in col_lower for kw in ["profit", "center", "pc", "department", "unit", "branch"]
            ):
                schema["profit_center"] = col
                continue

            if not schema["categorical_segment"] and any(
                kw in col_lower for kw in ["broker", "region", "product", "lob", "type", "segment", "state"]
            ):
                schema["categorical_segment"] = col
                continue

            if not schema["client_id"] and any(
                kw in col_lower for kw in ["id", "client", "customer", "account", "name", "policy"]
            ):
                schema["client_id"] = col

        if not schema["financial_metric"]:
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                schema["financial_metric"] = num_cols[0]

        if not schema["timeline_metric"]:
            date_cols = self.df.select_dtypes(include=["datetime", "object"]).columns
            if len(date_cols) > 0:
                schema["timeline_metric"] = date_cols[0]

        if not schema["profit_center"]:
            cat_cols = self.df.select_dtypes(include=["object"]).columns
            if len(cat_cols) > 1:
                schema["profit_center"] = cat_cols[1]

        if not schema["categorical_segment"]:
            cat_cols = self.df.select_dtypes(include=["object"]).columns
            if len(cat_cols) > 0:
                schema["categorical_segment"] = cat_cols[0]

        return {
            "columns": columns,
            "inferred_mapping": schema,
            "profit_centers": self.get_profit_centers(schema["profit_center"])[:100],
            "agency_codes": self.get_agency_codes(schema["agency_code"])[:500],
            "date_range": self.get_date_range(schema["timeline_metric"]),
            "baseline_date": ANALYTICAL_BASELINE.strftime("%Y-%m-%d"),
            "has_business_type": schema["business_type"] is not None,
            "data_quality": self.assess_data_quality(schema)
        }

    def assess_data_quality(self, mapping: dict) -> dict:
        """
        Surface data quality issues in the mapped columns before analysis runs, so problems
        like blank premiums or unparseable dates are visible instead of silently dropped.
        """
        total_rows = int(len(self.df))
        warnings = []

        report = {
            "total_rows": total_rows,
            "duplicate_rows": int(self.df.duplicated().sum()) if total_rows > 0 else 0,
            "financial_metric": None,
            "timeline_metric": None,
            "optional_columns": {},
            "warnings": warnings
        }

        if total_rows == 0:
            warnings.append("The uploaded file has no data rows.")
            return report

        if report["duplicate_rows"] > 0:
            dup_pct = report["duplicate_rows"] / total_rows * 100
            warnings.append(f"{report['duplicate_rows']:,} fully duplicate rows found ({dup_pct:.1f}% of the file).")

        fin_col = mapping.get("financial_metric")

        if fin_col and fin_col in self.df.columns:
            raw = self.df[fin_col]
            numeric = pd.to_numeric(raw, errors="coerce")
            missing_pct = float(numeric.isna().mean() * 100)
            non_positive_pct = float((numeric.fillna(0) <= 0).mean() * 100)

            report["financial_metric"] = {
                "column": fin_col,
                "missing_or_non_numeric_pct": round(missing_pct, 1),
                "zero_or_negative_pct": round(non_positive_pct, 1)
            }

            if missing_pct >= 5:
                warnings.append(f"'{fin_col}' has {missing_pct:.1f}% blank or non-numeric values.")
            if non_positive_pct >= 5:
                warnings.append(f"'{fin_col}' has {non_positive_pct:.1f}% zero or negative values.")

        time_col = mapping.get("timeline_metric")

        if time_col and time_col in self.df.columns:
            raw = self.df[time_col]
            parsed = pd.to_datetime(raw, errors="coerce")
            unparseable_pct = float(parsed.isna().mean() * 100)

            valid_parsed = parsed.dropna()
            before_baseline_pct = float((valid_parsed < ANALYTICAL_BASELINE).mean() * 100) if not valid_parsed.empty else 0.0
            today = pd.Timestamp(datetime.now().date())
            future_pct = float((valid_parsed > today).mean() * 100) if not valid_parsed.empty else 0.0

            report["timeline_metric"] = {
                "column": time_col,
                "unparseable_pct": round(unparseable_pct, 1),
                "before_baseline_pct": round(before_baseline_pct, 1),
                "future_dated_pct": round(future_pct, 1)
            }

            if unparseable_pct >= 2:
                warnings.append(f"'{time_col}' has {unparseable_pct:.1f}% unparseable dates that will be excluded.")
            if before_baseline_pct >= 5:
                warnings.append(
                    f"'{time_col}' has {before_baseline_pct:.1f}% of dates before {ANALYTICAL_BASELINE.strftime('%Y-%m-%d')} "
                    f"(outside the analytical baseline and excluded)."
                )
            if future_pct >= 5:
                warnings.append(f"'{time_col}' has {future_pct:.1f}% future-dated records.")

        for key in ("profit_center", "agency_code", "categorical_segment", "client_id", "business_type"):
            col = mapping.get(key)

            if not col or col not in self.df.columns:
                continue

            missing_pct = float(self.df[col].isna().mean() * 100)
            report["optional_columns"][key] = {
                "column": col,
                "missing_pct": round(missing_pct, 1)
            }

            if missing_pct >= 10:
                warnings.append(f"'{col}' ({key.replace('_', ' ')}) is {missing_pct:.1f}% blank.")

        return report

    def _empty_forecast_outlook(self, projection_target="premium"):
        return {
            "metric_type": projection_target,
            "current_year": None,
            "previous_year": None,
            "current_actual": 0.0,
            "previous_year_actual": 0.0,
            "projected_year_end": 0.0,
            "conservative_year_end": 0.0,
            "aggressive_year_end": 0.0,
            "remaining_months": 0,
            "growth_vs_previous_year_pct": 0.0,
            "confidence_score": 0.0,
            "confidence_label": "Insufficient Data",
            "trend_direction": "Flat",
            "monthly_forecast": [],
            "seasonal_index": {},
            "executive_summary": "Not enough data is available to produce a reliable year-end projection."
        }

    def _build_monthly_series(self, df, time_col, fin_col, id_col, target_series="premium"):
        """Aggregate a (already scoped/filtered) dataframe into a monthly time series."""
        if df is None or df.empty or time_col not in df.columns:
            return pd.DataFrame(columns=["YearMonth", "premium", "count"])

        d = df.copy()
        d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
        d = d.dropna(subset=[time_col])

        if d.empty:
            return pd.DataFrame(columns=["YearMonth", "premium", "count"])

        d["YearMonth"] = d[time_col].dt.to_period("M")
        grouped = d.groupby("YearMonth")

        monthly = pd.DataFrame({
            "premium": grouped[fin_col].sum() if fin_col in d.columns else grouped.size(),
            "count": grouped[id_col].nunique() if id_col and id_col in d.columns else grouped.size()
        }).reset_index()

        return monthly.sort_values("YearMonth").reset_index(drop=True)

    def _seasonal_trend_forecast(self, monthly_df, target_series, periods_ahead):
        """
        Produce a trend + seasonality aware forecast for the next `periods_ahead` months.

        Method: fit a linear trend across all available months, derive a per-calendar-month
        seasonal multiplier from the average ratio of actual-to-trend, normalize the
        multipliers to average 1.0, then project future months as trend * seasonal index.
        Confidence bands are built from the standard deviation of de-seasonalized residuals.
        """
        empty_diagnostics = {"r_squared": 0.0, "residual_std": 0.0, "slope": 0.0, "seasonal_index": {}}

        if monthly_df is None or monthly_df.empty or len(monthly_df) < 2 or periods_ahead <= 0:
            return [], empty_diagnostics

        mdf = monthly_df.copy().sort_values("YearMonth").reset_index(drop=True)
        mdf["CalMonth"] = mdf["YearMonth"].dt.month

        X = np.arange(len(mdf)).reshape(-1, 1)
        y = mdf[target_series].astype(float).values

        model = LinearRegression().fit(X, y)
        trend_fitted = model.predict(X)

        try:
            r_squared = float(model.score(X, y))
        except Exception:
            r_squared = 0.0

        safe_trend = np.where(np.abs(trend_fitted) < 1e-9, np.nan, trend_fitted)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = y / safe_trend

        mdf["_ratio"] = ratio

        seasonal_index = {}
        for m in range(1, 13):
            vals = mdf.loc[mdf["CalMonth"] == m, "_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
            seasonal_index[m] = float(vals.mean()) if len(vals) > 0 else 1.0

        observed_vals = [v for v in seasonal_index.values() if v is not None and not np.isnan(v)]
        mean_idx = float(np.mean(observed_vals)) if observed_vals else 1.0

        if mean_idx and not np.isnan(mean_idx) and mean_idx != 0:
            seasonal_index = {m: v / mean_idx for m, v in seasonal_index.items()}

        # Clip extreme seasonal swings so sparse-history months don't produce wild forecasts.
        seasonal_index = {m: float(np.clip(v, 0.4, 2.5)) for m, v in seasonal_index.items()}

        deseasonalized_fitted = trend_fitted * np.array([seasonal_index.get(m, 1.0) for m in mdf["CalMonth"]])
        residuals = y - deseasonalized_fitted
        residual_std = float(np.std(residuals)) if len(residuals) > 1 else 0.0

        last_period = mdf["YearMonth"].max()
        future_X = np.arange(len(mdf), len(mdf) + periods_ahead).reshape(-1, 1)
        future_trend = model.predict(future_X)

        future = []

        for i, base_pred in enumerate(future_trend):
            future_period = last_period + i + 1
            cal_month = int(future_period.month)
            seasonal_mult = seasonal_index.get(cal_month, 1.0)
            expected = max(0.0, float(base_pred) * seasonal_mult)
            conservative = max(0.0, expected - residual_std)
            aggressive = max(0.0, expected + residual_std)

            future.append({
                "period": str(future_period),
                "year": int(future_period.year),
                "month": cal_month,
                "expected_value": expected,
                "conservative_value": conservative,
                "aggressive_value": aggressive
            })

        slope = float(model.coef_[0]) if hasattr(model, "coef_") and len(model.coef_) > 0 else 0.0

        return future, {
            "r_squared": r_squared,
            "residual_std": residual_std,
            "slope": slope,
            "seasonal_index": seasonal_index
        }

    def _empty_ai_insights(self):
        return {
            "portfolio_health_score": 0.0,
            "portfolio_health_label": "Insufficient Data",
            "portfolio_health_status": "neutral",
            "executive_summary": "Not enough data is available to generate AI insights.",
            "insights": [],
            "recommended_actions": [
                "Upload or select a broader dataset to generate portfolio insights."
            ]
        }

    def _period_bounds(self, period: str, anchor: pd.Timestamp):
        """Return (start, end) timestamps for the annual/quarterly/monthly window containing `anchor`."""
        anchor = pd.Timestamp(anchor)

        if period == "monthly":
            start = pd.Timestamp(year=anchor.year, month=anchor.month, day=1)
            end = start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
        elif period == "quarterly":
            q_start_month = ((anchor.month - 1) // 3) * 3 + 1
            start = pd.Timestamp(year=anchor.year, month=q_start_month, day=1)
            end = start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        else:
            start = pd.Timestamp(year=anchor.year, month=1, day=1)
            end = pd.Timestamp(year=anchor.year, month=12, day=31)

        return start, end

    def _apply_goal_scope(self, df, scope_type, scope_value, pc_col, agency_col, cat_col):
        """Filter a dataframe down to the dimension a single goal targets."""
        if scope_type == "profit_center" and pc_col and pc_col in df.columns:
            normalized = self._normalize_categorical_value(scope_value)
            return df[df[pc_col] == normalized]

        if scope_type == "agency_code" and agency_col and agency_col in df.columns:
            normalized = self._normalize_categorical_value(scope_value)
            return df[df[agency_col] == normalized]

        if scope_type == "segment" and cat_col and cat_col in df.columns:
            return df[df[cat_col].astype(str) == str(scope_value)]

        return df

    def compute_goals(
        self,
        working_df,
        fin_col,
        id_col,
        time_col,
        pc_col,
        agency_col,
        cat_col,
        projection_target,
        goals,
        anchor_date
    ) -> list:
        """
        Compute pacing + a seasonality-aware projection for each goal in `goals`.

        Each goal dict supports: id, label, period ("annual"/"quarterly"/"monthly"),
        scope_type ("overall"/"profit_center"/"agency_code"/"segment"), scope_value, target_value.
        """
        target_series = "premium" if projection_target == "premium" else "count"
        results = []

        if not goals:
            return results

        if anchor_date is None or pd.isna(anchor_date):
            anchor_date = pd.Timestamp(datetime.now().date())

        for idx, g in enumerate(goals):
            goal_id = g.get("id") or f"goal_{idx + 1}"
            label = g.get("label") or "Goal"
            period = g.get("period") if g.get("period") in ("annual", "quarterly", "monthly") else "annual"
            scope_type = g.get("scope_type") if g.get("scope_type") in (
                "overall", "profit_center", "agency_code", "segment"
            ) else "overall"
            scope_value = g.get("scope_value")

            try:
                target_value = float(g.get("target_value") or 0)
            except (ValueError, TypeError):
                target_value = 0.0

            start, end = self._period_bounds(period, anchor_date)
            base_result = {
                "id": goal_id,
                "label": label,
                "period": period,
                "scope_type": scope_type,
                "scope_value": scope_value,
                "metric_type": projection_target,
                "period_start": start.strftime("%Y-%m-%d"),
                "period_end": end.strftime("%Y-%m-%d"),
                "days_total": int((end - start).days + 1)
            }

            scoped_df = self._apply_goal_scope(working_df, scope_type, scope_value, pc_col, agency_col, cat_col)

            if target_value <= 0:
                results.append({
                    **base_result,
                    "target": 0.0,
                    "actual": 0.0,
                    "achievement_pct": 0.0,
                    "expected_pct": 0.0,
                    "gap_to_goal": 0.0,
                    "projected_period_end": 0.0,
                    "projected_period_end_low": None,
                    "projected_period_end_high": None,
                    "days_elapsed": 0,
                    "status": "no_goal_set",
                    "trend": []
                })
                continue

            effective_end_for_actual = min(end, anchor_date)
            in_period = (scoped_df[time_col] >= start) & (scoped_df[time_col] <= effective_end_for_actual)
            period_df = scoped_df[in_period]

            if projection_target == "premium":
                actual = float(period_df[fin_col].sum()) if not period_df.empty else 0.0
            else:
                actual = float(period_df[id_col].nunique()) if id_col and not period_df.empty else float(len(period_df))

            days_total = max(1, base_result["days_total"])
            days_elapsed = max(0, (effective_end_for_actual - start).days + 1) if effective_end_for_actual >= start else 0

            expected_pct = min(100.0, (days_elapsed / days_total) * 100)
            achievement_pct = (actual / target_value) * 100
            gap_to_goal = float(actual - target_value)

            projected_period_end = float((actual / days_elapsed) * days_total) if days_elapsed > 0 else 0.0
            projected_period_end_low = None
            projected_period_end_high = None

            # Where at least one full calendar month remains in the goal period, refine the
            # run-rate projection using the seasonality-aware monthly forecast instead.
            anchor_month_period = pd.Period(anchor_date, freq="M")
            end_month_period = pd.Period(end, freq="M")

            if anchor_month_period < end_month_period:
                remaining_months = int(end_month_period.ordinal - anchor_month_period.ordinal)
                monthly_series = self._build_monthly_series(scoped_df, time_col, fin_col, id_col, target_series)

                if len(monthly_series) >= 2:
                    future_vals, _diag = self._seasonal_trend_forecast(monthly_series, target_series, remaining_months)
                    remaining_forecast = sum(item["expected_value"] for item in future_vals)
                    remaining_conservative = sum(item["conservative_value"] for item in future_vals)
                    remaining_aggressive = sum(item["aggressive_value"] for item in future_vals)

                    projected_period_end = float(actual + remaining_forecast)
                    projected_period_end_low = float(actual + remaining_conservative)
                    projected_period_end_high = float(actual + remaining_aggressive)

            if expected_pct <= 0:
                status = "no_time_elapsed"
            else:
                pace_ratio = achievement_pct / expected_pct

                if pace_ratio >= 1.0:
                    status = "ahead"
                elif pace_ratio >= 0.8:
                    status = "on_pace"
                else:
                    status = "behind"

            trend = self._build_goal_trend(
                scoped_df, time_col, fin_col, id_col, projection_target, start, effective_end_for_actual, days_total
            )

            results.append({
                **base_result,
                "target": float(target_value),
                "actual": float(actual),
                "achievement_pct": float(achievement_pct),
                "expected_pct": float(expected_pct),
                "gap_to_goal": float(gap_to_goal),
                "projected_period_end": float(projected_period_end),
                "projected_period_end_low": projected_period_end_low,
                "projected_period_end_high": projected_period_end_high,
                "days_elapsed": int(days_elapsed),
                "status": status,
                "trend": trend
            })

        return results

    def _build_goal_trend(self, scoped_df, time_col, fin_col, id_col, projection_target, start, anchor, days_total, max_points: int = 10) -> list:
        """
        Build a downsampled cumulative-actual-vs-expected-pace series for a goal's sparkline,
        covering from the period start through the anchor date (today, or the data's latest date).
        """
        if scoped_df is None or scoped_df.empty or anchor < start:
            return []

        mask = (scoped_df[time_col] >= start) & (scoped_df[time_col] <= anchor)
        subset = scoped_df.loc[mask]

        if subset.empty:
            return []

        day_index = subset[time_col].dt.date

        if projection_target == "premium":
            daily = subset.groupby(day_index)[fin_col].sum().sort_index()
            cumulative = daily.cumsum()
        elif id_col and id_col in subset.columns:
            first_seen_in_period = subset.groupby(id_col)[time_col].min()
            daily_new = first_seen_in_period.dt.date.value_counts().sort_index()
            cumulative = daily_new.cumsum()
        else:
            daily = subset.groupby(day_index).size().sort_index()
            cumulative = daily.cumsum()

        if cumulative.empty:
            return []

        dates = cumulative.index.to_list()
        values = cumulative.values.tolist()
        total_points = len(values)
        n_points = min(max_points, total_points)

        if n_points <= 1:
            sample_positions = [total_points - 1]
        else:
            sample_positions = sorted(set(
                round(i * (total_points - 1) / (n_points - 1)) for i in range(n_points)
            ))

        trend = []

        for pos in sample_positions:
            elapsed_days = (dates[pos] - start.date()).days + 1
            expected_pct = min(100.0, (elapsed_days / max(1, days_total)) * 100)

            trend.append({
                "date": dates[pos].strftime("%Y-%m-%d"),
                "cumulative_actual": float(values[pos]),
                "expected_pct_at_date": float(expected_pct)
            })

        return trend

    # ----------------------------------------------------------------------- #
    # Goal suggestions ("based on past info, here's what to aim for")
    # ----------------------------------------------------------------------- #

    def _reference_period_bounds(self, period: str, anchor: pd.Timestamp):
        """The same period one year prior — the historical baseline a suggested goal is built from."""
        start, end = self._period_bounds(period, anchor)
        return start - pd.DateOffset(years=1), end - pd.DateOffset(years=1)

    def _period_actual(self, df, time_col, fin_col, id_col, projection_target, start, end):
        if df is None or df.empty:
            return 0.0

        mask = (df[time_col] >= start) & (df[time_col] <= end)
        subset = df[mask]

        if subset.empty:
            return 0.0

        if projection_target == "premium":
            return float(subset[fin_col].sum())

        return float(subset[id_col].nunique()) if id_col else float(len(subset))

    def _estimate_growth_rate(self, df, time_col, fin_col, id_col, projection_target, period, anchor_date):
        """
        Compare the most recent complete comparable period to the one a year before it
        (e.g. last calendar year vs the year before). Returns None if there isn't enough
        history for this scope to compute a meaningful rate.
        """
        ref_start, ref_end = self._reference_period_bounds(period, anchor_date)
        prior_start, prior_end = ref_start - pd.DateOffset(years=1), ref_end - pd.DateOffset(years=1)

        ref_actual = self._period_actual(df, time_col, fin_col, id_col, projection_target, ref_start, ref_end)
        prior_actual = self._period_actual(df, time_col, fin_col, id_col, projection_target, prior_start, prior_end)

        if prior_actual <= 0:
            return None

        growth = (ref_actual - prior_actual) / prior_actual

        # Clip so a single noisy small-scope period can't produce an absurd suggestion.
        return float(np.clip(growth, -0.6, 1.5))

    def _round_target(self, value: float) -> float:
        """Round a suggested target to a clean, presentable figure (e.g. 6,432,911 -> 6,400,000)."""
        if value is None or value <= 0:
            return 0.0

        digits = len(str(int(value)))
        magnitude = max(1, 10 ** max(0, digits - 2))

        return float(round(value / magnitude) * magnitude)

    def suggest_goal_candidates(
        self,
        mapping: dict,
        projection_target: str = "premium",
        period: str = "annual",
        business_view: str = "all",
        top_n: int = 3
    ) -> list:
        """
        Propose goal candidates from historical performance: the overall book plus the
        top N profit centers, agencies, and segments by volume. Each candidate reports last
        year's actual for the same period, the book's own trailing growth rate, and three
        target tiers (maintain / grow / stretch) derived from those two numbers.
        """
        fin_col = mapping.get("financial_metric")
        time_col = mapping.get("timeline_metric")
        cat_col = mapping.get("categorical_segment")
        pc_col = mapping.get("profit_center")
        id_col = mapping.get("client_id")
        agency_col = mapping.get("agency_code")
        biz_type_col = mapping.get("business_type")

        if not fin_col or not time_col or fin_col not in self.df.columns or time_col not in self.df.columns:
            raise ValueError("Financial and Timeline columns must be mapped and present to suggest goals.")

        cat_col = cat_col if cat_col in self.df.columns else None
        pc_col = pc_col if pc_col in self.df.columns else None
        id_col = id_col if id_col in self.df.columns else None
        agency_col = agency_col if agency_col in self.df.columns else None
        biz_type_col = biz_type_col if biz_type_col in self.df.columns else None

        cols = list(dict.fromkeys(
            [fin_col, time_col] + [c for c in [cat_col, pc_col, id_col, agency_col, biz_type_col] if c]
        ))

        working_df = self.df[cols].copy()
        working_df[time_col] = pd.to_datetime(working_df[time_col], errors="coerce")
        working_df[fin_col] = pd.to_numeric(working_df[fin_col], errors="coerce").fillna(0)
        working_df = working_df.dropna(subset=[time_col])
        working_df = working_df[working_df[time_col] >= ANALYTICAL_BASELINE]

        if pc_col:
            working_df[pc_col] = working_df[pc_col].apply(self._normalize_categorical_value)
        if agency_col:
            working_df[agency_col] = working_df[agency_col].apply(self._normalize_categorical_value)
        if id_col:
            working_df[id_col] = working_df[id_col].fillna("Unknown")

        if working_df.empty:
            return []

        anchor_date = working_df[time_col].max()

        working_df = self._assign_business_type(working_df, id_col, time_col, biz_type_col)

        if business_view == "new":
            working_df = working_df[working_df["BusinessType"] == "New"]
        elif business_view == "renewal":
            working_df = working_df[working_df["BusinessType"] == "Renewal"]

        if working_df.empty:
            return []

        # Organization-wide growth rate, used as a fallback for scopes with too little
        # standalone history (e.g. a small agency with only a few months of data).
        overall_growth = self._estimate_growth_rate(
            working_df, time_col, fin_col, id_col, projection_target, period, anchor_date
        )

        scope_candidates = [("overall", None, "Overall Book")]

        def top_values(col):
            if not col or col not in working_df.columns:
                return []

            sums = working_df.groupby(col)[fin_col].sum().sort_values(ascending=False)
            return list(sums.head(top_n).index)

        for val in top_values(pc_col):
            scope_candidates.append(("profit_center", val, f"Profit Center: {val}"))
        for val in top_values(agency_col):
            scope_candidates.append(("agency_code", val, f"Agency: {val}"))
        for val in top_values(cat_col):
            scope_candidates.append(("segment", val, f"Segment: {val}"))

        suggestions = []

        for scope_type, scope_value, scope_label in scope_candidates:
            scoped_df = self._apply_goal_scope(working_df, scope_type, scope_value, pc_col, agency_col, cat_col)

            ref_start, ref_end = self._reference_period_bounds(period, anchor_date)
            cur_start, cur_end = self._period_bounds(period, anchor_date)

            reference_actual = self._period_actual(
                scoped_df, time_col, fin_col, id_col, projection_target, ref_start, ref_end
            )

            if reference_actual <= 0:
                continue

            growth = self._estimate_growth_rate(
                scoped_df, time_col, fin_col, id_col, projection_target, period, anchor_date
            )
            data_sufficient = growth is not None

            if growth is None:
                growth = overall_growth if overall_growth is not None else 0.0

            suggestions.append({
                "scope_type": scope_type,
                "scope_value": scope_value,
                "scope_label": scope_label,
                "period": period,
                "period_start": cur_start.strftime("%Y-%m-%d"),
                "period_end": cur_end.strftime("%Y-%m-%d"),
                "reference_period_start": ref_start.strftime("%Y-%m-%d"),
                "reference_period_end": ref_end.strftime("%Y-%m-%d"),
                "reference_actual": float(reference_actual),
                "growth_rate_pct": float(growth * 100),
                "data_sufficient": data_sufficient,
                "metric_type": projection_target,
                "suggestions": {
                    "maintain": self._round_target(reference_actual),
                    "grow": self._round_target(reference_actual * (1 + growth)),
                    "stretch": self._round_target(reference_actual * (1 + growth + 0.05))
                }
            })

        return suggestions

    def _assign_business_type(self, working_df, id_col, time_col, biz_type_col):
        if biz_type_col and biz_type_col in working_df.columns:
            working_df["BusinessType"] = working_df[biz_type_col].apply(self._classify_business_type)
            other_mask = working_df["BusinessType"] == "Other"

            if other_mask.any() and id_col and id_col in working_df.columns:
                first_seen = self.df.copy()
                first_seen[time_col] = pd.to_datetime(first_seen[time_col], errors="coerce")
                first_seen = first_seen.dropna(subset=[time_col])
                account_birthdays = first_seen.groupby(id_col)[time_col].min()

                def derive(row):
                    acc = row[id_col]

                    if acc in account_birthdays.index:
                        birthday = account_birthdays[acc]

                        if pd.notna(row[time_col]) and abs((row[time_col] - birthday).days) <= 30:
                            return "New"

                        return "Renewal"

                    return "Renewal"

                working_df.loc[other_mask, "BusinessType"] = working_df.loc[other_mask].apply(derive, axis=1)

        elif id_col and id_col in working_df.columns:
            first_seen = self.df.copy()
            first_seen[time_col] = pd.to_datetime(first_seen[time_col], errors="coerce")
            first_seen = first_seen.dropna(subset=[time_col])
            account_birthdays = first_seen.groupby(id_col)[time_col].min()

            def classify_row(row):
                acc = row[id_col]

                if acc in account_birthdays.index:
                    birthday = account_birthdays[acc]

                    if pd.notna(row[time_col]) and abs((row[time_col] - birthday).days) <= 30:
                        return "New"

                    return "Renewal"

                return "Renewal"

            working_df["BusinessType"] = working_df.apply(classify_row, axis=1)

        else:
            working_df["BusinessType"] = "Renewal"

        return working_df

    def _compute_forecast_outlook(self, monthly_df, target_series, projection_target):
        if monthly_df is None or monthly_df.empty or len(monthly_df) < 2:
            return self._empty_forecast_outlook(projection_target)

        forecast_df = monthly_df.copy().sort_values("YearMonth").reset_index(drop=True)
        forecast_df["Year"] = forecast_df["YearMonth"].dt.year

        current_year = int(forecast_df["Year"].max())
        previous_year = current_year - 1

        current_year_df = forecast_df[forecast_df["Year"] == current_year]
        previous_year_df = forecast_df[forecast_df["Year"] == previous_year]

        current_actual = float(current_year_df[target_series].sum())
        previous_year_actual = float(previous_year_df[target_series].sum()) if not previous_year_df.empty else 0.0

        last_period = forecast_df["YearMonth"].max()
        remaining_months = max(0, 12 - int(last_period.month))

        future_monthly, diagnostics = self._seasonal_trend_forecast(forecast_df, target_series, remaining_months)

        r_squared = diagnostics["r_squared"]
        residual_std = diagnostics["residual_std"]
        slope = diagnostics["slope"]

        avg_monthly = float(forecast_df[target_series].astype(float).mean()) if len(forecast_df) > 0 else 0.0
        volatility_ratio = abs(residual_std / avg_monthly) if avg_monthly != 0 else 1.0

        expected_future_total = sum(item["expected_value"] for item in future_monthly)
        conservative_future_total = sum(item["conservative_value"] for item in future_monthly)
        aggressive_future_total = sum(item["aggressive_value"] for item in future_monthly)

        projected_year_end = float(current_actual + expected_future_total)
        conservative_year_end = float(current_actual + conservative_future_total)
        aggressive_year_end = float(current_actual + aggressive_future_total)

        growth_vs_previous_year_pct = 0.0

        if previous_year_actual > 0:
            growth_vs_previous_year_pct = ((projected_year_end - previous_year_actual) / previous_year_actual) * 100

        if slope > 0:
            trend_direction = "Increasing"
        elif slope < 0:
            trend_direction = "Decreasing"
        else:
            trend_direction = "Flat"

        data_points = len(forecast_df)
        history_score = min(1.0, data_points / 12)
        fit_score = max(0.0, min(1.0, r_squared))
        stability_score = max(0.0, min(1.0, 1 - volatility_ratio))

        confidence_score = round(((history_score * 0.35) + (fit_score * 0.40) + (stability_score * 0.25)) * 100, 1)

        if data_points < 4:
            confidence_label = "Low"
        elif confidence_score >= 75:
            confidence_label = "High"
        elif confidence_score >= 50:
            confidence_label = "Moderate"
        else:
            confidence_label = "Low"

        metric_label = "premium volume" if projection_target == "premium" else "policy count"

        if previous_year_actual > 0:
            growth_phrase = f"{growth_vs_previous_year_pct:.1f}% compared with the prior year"
        else:
            growth_phrase = "no prior-year comparison is available"

        executive_summary = (
            f"Based on current monthly performance and seasonal patterns, the selected portfolio is projected to finish "
            f"{current_year} at approximately {projected_year_end:,.0f} in {metric_label}. "
            f"This represents {growth_phrase}. "
            f"The forecast confidence is {confidence_label.lower()} based on available history, trend fit, and volatility. "
            f"See the Goals panel for goal-specific pacing and projections."
        )

        return {
            "metric_type": projection_target,
            "current_year": current_year,
            "previous_year": previous_year,
            "current_actual": float(current_actual),
            "previous_year_actual": float(previous_year_actual),
            "projected_year_end": float(projected_year_end),
            "conservative_year_end": float(conservative_year_end),
            "aggressive_year_end": float(aggressive_year_end),
            "remaining_months": int(remaining_months),
            "growth_vs_previous_year_pct": float(growth_vs_previous_year_pct),
            "confidence_score": float(confidence_score),
            "confidence_label": confidence_label,
            "trend_direction": trend_direction,
            "monthly_forecast": future_monthly,
            "seasonal_index": diagnostics.get("seasonal_index", {}),
            "executive_summary": executive_summary
        }

    def _generate_ai_insights(
        self,
        total_premium,
        total_accounts,
        avg_account_size,
        retention_rate,
        hhi_index,
        pareto_ratio,
        business_split,
        forecast_outlook,
        primary_goal,
        segment_data,
        anomalies,
        projection_target
    ):
        insights = []
        action_items = []

        metric_label = "premium volume" if projection_target == "premium" else "policy count"

        projected_year_end = float(forecast_outlook.get("projected_year_end", 0) or 0)
        growth_pct = float(forecast_outlook.get("growth_vs_previous_year_pct", 0) or 0)
        confidence_label = forecast_outlook.get("confidence_label", "Insufficient Data")
        trend_direction = forecast_outlook.get("trend_direction", "Flat")

        goal_status = "No Goal Set"
        projected_gap_to_goal = 0.0

        if primary_goal and float(primary_goal.get("target", 0) or 0) > 0:
            projected_gap_to_goal = float(primary_goal.get("projected_period_end", 0) or 0) - float(primary_goal.get("target", 0) or 0)
            goal_status = "Projected Above Goal" if projected_gap_to_goal >= 0 else "Projected Below Goal"

        new_premium = float(business_split.get("new_business_premium", 0) or 0)
        renewal_premium = float(business_split.get("renewal_premium", 0) or 0)
        new_count = int(business_split.get("new_business_count", 0) or 0)
        renewal_count = int(business_split.get("renewal_count", 0) or 0)

        total_split_premium = new_premium + renewal_premium
        total_split_count = new_count + renewal_count

        new_premium_share = (new_premium / total_split_premium * 100) if total_split_premium > 0 else 0
        renewal_premium_share = (renewal_premium / total_split_premium * 100) if total_split_premium > 0 else 0
        new_count_share = (new_count / total_split_count * 100) if total_split_count > 0 else 0
        renewal_count_share = (renewal_count / total_split_count * 100) if total_split_count > 0 else 0

        top_segment_name = None
        top_segment_share = 0.0

        if segment_data and len(segment_data) > 0:
            try:
                top_segment_name = max(segment_data, key=segment_data.get)
                segment_total = sum(float(v or 0) for v in segment_data.values())

                if segment_total > 0:
                    top_segment_share = float(segment_data.get(top_segment_name, 0) or 0) / segment_total * 100
            except Exception:
                top_segment_name = None
                top_segment_share = 0.0

        anomaly_count = len(anomalies) if anomalies else 0

        if hhi_index >= 2500:
            concentration_level = "High"
        elif hhi_index >= 1500:
            concentration_level = "Moderate"
        else:
            concentration_level = "Low"

        if retention_rate >= 90:
            retention_level = "Strong"
        elif retention_rate >= 75:
            retention_level = "Watch"
        else:
            retention_level = "At Risk"

        if growth_pct >= 10:
            growth_level = "Accelerating"
        elif growth_pct >= 3:
            growth_level = "Growing"
        elif growth_pct <= -5:
            growth_level = "Declining"
        else:
            growth_level = "Flat"

        if projected_year_end > 0:
            insights.append({
                "category": "Forecast",
                "icon": "🔮",
                "severity": "positive" if growth_pct >= 0 else "warning",
                "title": "Year-End Projection",
                "message": (
                    f"Based on current monthly performance, the selected book is projected to finish at "
                    f"approximately {projected_year_end:,.0f} in {metric_label}. Forecast confidence is "
                    f"{confidence_label.lower()}."
                )
            })
        else:
            insights.append({
                "category": "Forecast",
                "icon": "🔮",
                "severity": "neutral",
                "title": "Forecast Availability",
                "message": "There is not enough monthly history available to produce a reliable year-end forecast."
            })

        if growth_level == "Accelerating":
            insights.append({
                "category": "Growth",
                "icon": "📈",
                "severity": "positive",
                "title": "Growth Momentum Is Strong",
                "message": (
                    f"The forecast indicates accelerating growth of approximately {growth_pct:.1f}% versus the prior year. "
                    f"Current trend direction is {trend_direction.lower()}."
                )
            })
            action_items.append(
                "Review the highest-performing segments to identify where growth is coming from and whether it can be replicated."
            )
        elif growth_level == "Growing":
            insights.append({
                "category": "Growth",
                "icon": "📈",
                "severity": "positive",
                "title": "Growth Trend Is Positive",
                "message": (
                    f"The book is projected to grow by approximately {growth_pct:.1f}% versus the prior year, "
                    f"suggesting positive but controlled expansion."
                )
            })
        elif growth_level == "Declining":
            insights.append({
                "category": "Growth",
                "icon": "📈",
                "severity": "risk",
                "title": "Growth Trend Is Declining",
                "message": (
                    f"The forecast indicates a decline of approximately {abs(growth_pct):.1f}% versus the prior year. "
                    f"This may require review of lost volume, renewal pressure, or reduced new business activity."
                )
            })
            action_items.append(
                "Investigate whether the decline is concentrated in specific profit centers, agencies, segments, or renewal groups."
            )
        else:
            insights.append({
                "category": "Growth",
                "icon": "📈",
                "severity": "neutral",
                "title": "Growth Trend Is Relatively Flat",
                "message": (
                    f"The current forecast shows limited movement versus the prior year at approximately {growth_pct:.1f}%."
                )
            })

        if goal_status == "Projected Above Goal":
            insights.append({
                "category": "Goal",
                "icon": "🏁",
                "severity": "positive",
                "title": "Projected Above Goal",
                "message": (
                    f"Current trends suggest the book may finish above goal by approximately "
                    f"{abs(projected_gap_to_goal):,.0f}."
                )
            })
        elif goal_status == "Projected Below Goal":
            insights.append({
                "category": "Goal",
                "icon": "🏁",
                "severity": "risk",
                "title": "Projected Below Goal",
                "message": (
                    f"Current trends suggest the book may finish below goal by approximately "
                    f"{abs(projected_gap_to_goal):,.0f}."
                )
            })
            action_items.append(
                "Compare the required monthly pace to recent monthly performance to determine whether the gap is realistically recoverable."
            )
        else:
            insights.append({
                "category": "Goal",
                "icon": "🏁",
                "severity": "neutral",
                "title": "No Goal Applied",
                "message": "No annual goal is currently applied, so goal-based variance is not being evaluated."
            })

        if retention_level == "Strong":
            insights.append({
                "category": "Retention",
                "icon": "✅",
                "severity": "positive",
                "title": "Retention Is Strong",
                "message": (
                    f"Account retention is currently {retention_rate:.1f}%, indicating healthy persistency across the selected book."
                )
            })
        elif retention_level == "Watch":
            insights.append({
                "category": "Retention",
                "icon": "✅",
                "severity": "warning",
                "title": "Retention Should Be Watched",
                "message": (
                    f"Account retention is currently {retention_rate:.1f}%. This is not critical, but it may deserve monitoring."
                )
            })
            action_items.append(
                "Look at renewal accounts by segment or agency to identify where retention is softening."
            )
        else:
            insights.append({
                "category": "Retention",
                "icon": "✅",
                "severity": "risk",
                "title": "Retention Risk Detected",
                "message": (
                    f"Account retention is currently {retention_rate:.1f}%, which may indicate elevated book persistence risk."
                )
            })
            action_items.append(
                "Prioritize reviewing accounts that were active in the prior year but are not appearing in the current year."
            )

        if concentration_level == "High":
            insights.append({
                "category": "Risk",
                "icon": "⚠️",
                "severity": "risk",
                "title": "High Concentration Risk",
                "message": (
                    f"The HHI concentration index is {hhi_index:,.0f}, which suggests elevated concentration exposure. "
                    f"{anomaly_count} concentration outlier account(s) were detected."
                )
            })
            action_items.append(
                "Review the largest accounts and determine whether the book is overly dependent on a small number of high-premium relationships."
            )
        elif concentration_level == "Moderate":
            insights.append({
                "category": "Risk",
                "icon": "⚠️",
                "severity": "warning",
                "title": "Moderate Concentration Risk",
                "message": (
                    f"The HHI concentration index is {hhi_index:,.0f}, suggesting moderate concentration. "
                    f"This is manageable, but still worth monitoring."
                )
            })
        else:
            insights.append({
                "category": "Risk",
                "icon": "⚠️",
                "severity": "positive",
                "title": "Concentration Appears Controlled",
                "message": (
                    f"The HHI concentration index is {hhi_index:,.0f}, suggesting the selected book is not overly concentrated."
                )
            })

        if total_split_premium > 0:
            if new_premium_share >= 35:
                insights.append({
                    "category": "Business Mix",
                    "icon": "🧭",
                    "severity": "positive",
                    "title": "New Business Contribution Is Strong",
                    "message": (
                        f"New business represents approximately {new_premium_share:.1f}% of premium volume, "
                        f"indicating strong contribution from new account activity."
                    )
                })
            elif new_premium_share >= 15:
                insights.append({
                    "category": "Business Mix",
                    "icon": "🧭",
                    "severity": "neutral",
                    "title": "Book Mix Is Renewal-Led With Meaningful New Business",
                    "message": (
                        f"Renewals represent approximately {renewal_premium_share:.1f}% of premium volume, "
                        f"while new business contributes {new_premium_share:.1f}%."
                    )
                })
            else:
                insights.append({
                    "category": "Business Mix",
                    "icon": "🧭",
                    "severity": "warning",
                    "title": "Book Is Heavily Renewal Dependent",
                    "message": (
                        f"New business represents only {new_premium_share:.1f}% of premium volume. "
                        f"The selected book appears highly dependent on renewals."
                    )
                })
                action_items.append(
                    "Review whether new business production is sufficient to offset future attrition risk."
                )
        elif total_split_count > 0:
            insights.append({
                "category": "Business Mix",
                "icon": "🧭",
                "severity": "neutral",
                "title": "Business Mix Available By Policy Count",
                "message": (
                    f"New business represents approximately {new_count_share:.1f}% of policies, while renewals represent "
                    f"{renewal_count_share:.1f}%."
                )
            })

        if top_segment_name:
            insights.append({
                "category": "Opportunity",
                "icon": "🎯",
                "severity": "positive" if top_segment_share >= 20 else "neutral",
                "title": "Largest Segment Opportunity",
                "message": (
                    f"{top_segment_name} is the largest visible segment in the selected scope, representing approximately "
                    f"{top_segment_share:.1f}% of measured volume. This segment may be useful for deeper opportunity review."
                )
            })

            if top_segment_share >= 35:
                action_items.append(
                    f"Evaluate whether {top_segment_name} concentration is strategic strength or a dependency risk."
                )

        if pareto_ratio <= 10:
            insights.append({
                "category": "Risk",
                "icon": "⚠️",
                "severity": "risk",
                "title": "Pareto Dependency Is Elevated",
                "message": (
                    f"Approximately {pareto_ratio:.1f}% of accounts appear to drive 80% of selected premium volume, "
                    f"which suggests a concentrated dependency profile."
                )
            })
        elif pareto_ratio <= 25:
            insights.append({
                "category": "Risk",
                "icon": "⚠️",
                "severity": "warning",
                "title": "Pareto Distribution Is Moderately Concentrated",
                "message": (
                    f"Approximately {pareto_ratio:.1f}% of accounts appear to drive 80% of selected premium volume."
                )
            })

        score = 100.0

        if growth_level == "Declining":
            score -= 20
        elif growth_level == "Flat":
            score -= 8

        if retention_level == "Watch":
            score -= 10
        elif retention_level == "At Risk":
            score -= 25

        if concentration_level == "Moderate":
            score -= 10
        elif concentration_level == "High":
            score -= 22

        if goal_status == "Projected Below Goal":
            score -= 15

        if confidence_label == "Low":
            score -= 8

        score = max(0.0, min(100.0, score))

        if score >= 85:
            health_label = "Excellent"
            health_status = "positive"
        elif score >= 70:
            health_label = "Healthy"
            health_status = "positive"
        elif score >= 55:
            health_label = "Watch"
            health_status = "warning"
        else:
            health_label = "At Risk"
            health_status = "risk"

        overview_parts = [
            f"The selected book is currently rated {health_label} with a portfolio health score of {score:.1f}/100."
        ]

        if projected_year_end > 0:
            overview_parts.append(
                f"The forecasted year-end position is approximately {projected_year_end:,.0f} in {metric_label}."
            )

        overview_parts.append(
            f"Retention is {retention_rate:.1f}% and concentration risk is classified as {concentration_level.lower()}."
        )

        if goal_status != "No Goal Set":
            overview_parts.append(f"Goal status is currently {goal_status.lower()}.")

        executive_summary = " ".join(overview_parts)

        if not action_items:
            action_items.append(
                "Continue monitoring forecast, retention, concentration, and new business mix as additional monthly data becomes available."
            )

        return {
            "portfolio_health_score": float(score),
            "portfolio_health_label": health_label,
            "portfolio_health_status": health_status,
            "executive_summary": executive_summary,
            "insights": insights,
            "recommended_actions": action_items[:5]
        }

    def _detect_trend_anomalies(
        self, working_df, time_col, fin_col, id_col, target_series, dimension_col, dimension_label, avg_monthly_scope_total
    ) -> list:
        """
        Flag entities (profit centers / agencies) whose most recent month deviates sharply
        from their own trailing 3-month average — a spike or drop worth a human look, distinct
        from the static "largest accounts" concentration check above.

        `avg_monthly_scope_total` is the whole book's average monthly volume, used to filter
        out entities too small to be worth flagging (compared like-for-like against a typical
        month, not the multi-year cumulative total).
        """
        flags = []

        if not dimension_col or dimension_col not in working_df.columns:
            return flags

        for value in working_df[dimension_col].dropna().unique():
            entity_df = working_df[working_df[dimension_col] == value]
            monthly = self._build_monthly_series(entity_df, time_col, fin_col, id_col, target_series)

            if len(monthly) < 4:
                continue

            monthly = monthly.sort_values("YearMonth").reset_index(drop=True)
            latest_value = float(monthly[target_series].iloc[-1])
            trailing_window = monthly[target_series].iloc[-4:-1]
            baseline = float(trailing_window.mean()) if len(trailing_window) > 0 else 0.0

            if baseline <= 0:
                continue

            # Ignore entities too small to matter relative to the book's typical month.
            if avg_monthly_scope_total > 0 and (baseline / avg_monthly_scope_total) < 0.03:
                continue

            deviation = (latest_value - baseline) / baseline

            if abs(deviation) >= 0.4:
                flags.append({
                    "scope_type": dimension_label,
                    "scope_value": str(value),
                    "latest_period": str(monthly["YearMonth"].iloc[-1]),
                    "latest_value": latest_value,
                    "trailing_avg": baseline,
                    "deviation_pct": float(deviation * 100),
                    "direction": "spike" if deviation > 0 else "drop"
                })

        return flags

    def run_analysis(
        self,
        mapping: dict,
        selected_profit_center: str = "ALL",
        projection_target: str = "premium",
        start_date: str = None,
        end_date: str = None,
        include_future_dates: bool = False,
        selected_agency_codes: list = None,
        goal_value: float = 0,
        goals: list = None,
        business_view: str = "all"
    ) -> dict:
        """
        Run the full book-of-business analysis pipeline: filter/scope the raw data per the
        request, compute KPIs, retention/concentration metrics, a seasonality-aware forecast,
        multi-goal pacing, and rule-based AI insights.

        Raises ValueError if the required financial/timeline columns are missing or not
        present in the uploaded file (e.g. a mapping left over from a previously loaded file).
        """
        if not isinstance(mapping, dict):
            raise ValueError("A column mapping is required to run analysis.")

        fin_col = mapping.get("financial_metric")
        time_col = mapping.get("timeline_metric")
        cat_col = mapping.get("categorical_segment")
        pc_col = mapping.get("profit_center")
        id_col = mapping.get("client_id")
        agency_col = mapping.get("agency_code")
        biz_type_col = mapping.get("business_type")

        if not fin_col or not time_col:
            raise ValueError("Financial and Timeline metrics are required fields.")

        if fin_col not in self.df.columns or time_col not in self.df.columns:
            raise ValueError(
                "The mapped Financial or Timeline column was not found in the uploaded file. "
                "This can happen if the mapping is left over from a different file — "
                "re-upload the file and confirm the schema mapping again."
            )

        # Optional mappings may reference a column that no longer exists (e.g. a stale mapping
        # or a saved view applied to a different file). Drop those silently rather than raising,
        # since the analysis can still run without them.
        cat_col = cat_col if cat_col in self.df.columns else None
        pc_col = pc_col if pc_col in self.df.columns else None
        id_col = id_col if id_col in self.df.columns else None
        agency_col = agency_col if agency_col in self.df.columns else None
        biz_type_col = biz_type_col if biz_type_col in self.df.columns else None

        cols_to_keep = [fin_col, time_col]

        if cat_col:
            cols_to_keep.append(cat_col)
        if pc_col:
            cols_to_keep.append(pc_col)
        if id_col:
            cols_to_keep.append(id_col)
        if agency_col:
            cols_to_keep.append(agency_col)
        if biz_type_col:
            cols_to_keep.append(biz_type_col)

        cols_to_keep = list(dict.fromkeys(cols_to_keep))

        working_df = self.df[cols_to_keep].copy()
        working_df[time_col] = pd.to_datetime(working_df[time_col], errors="coerce")
        working_df[fin_col] = pd.to_numeric(working_df[fin_col], errors="coerce").fillna(0)
        working_df = working_df.dropna(subset=[time_col])

        if pc_col:
            working_df[pc_col] = working_df[pc_col].apply(self._normalize_categorical_value)
        if agency_col:
            working_df[agency_col] = working_df[agency_col].apply(self._normalize_categorical_value)
        if cat_col:
            working_df[cat_col] = working_df[cat_col].fillna("Unknown")
        if id_col:
            working_df[id_col] = working_df[id_col].fillna("Unknown")

        working_df = working_df[working_df[time_col] >= ANALYTICAL_BASELINE]

        future_records_removed = 0
        future_dollar_amount = 0.0

        if not include_future_dates:
            today = pd.Timestamp(datetime.now().date())
            future_mask = working_df[time_col] > today
            future_records_removed = int(future_mask.sum())
            future_dollar_amount = float(working_df.loc[future_mask, fin_col].sum())
            working_df = working_df[~future_mask]

        effective_start = None
        effective_end = None

        if start_date:
            start_dt = pd.to_datetime(start_date, errors="coerce")

            if pd.notna(start_dt):
                working_df = working_df[working_df[time_col] >= start_dt]
                effective_start = start_dt

        if end_date:
            end_dt = pd.to_datetime(end_date, errors="coerce")

            if pd.notna(end_dt):
                end_capped = end_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)
                working_df = working_df[working_df[time_col] <= end_capped]
                effective_end = end_dt

        if effective_start is None and not working_df.empty:
            effective_start = working_df[time_col].min()

        if effective_end is None and not working_df.empty:
            effective_end = working_df[time_col].max()

        if pc_col and selected_profit_center and str(selected_profit_center).upper() != "ALL":
            normalized_selection = self._normalize_categorical_value(selected_profit_center)
            working_df = working_df[working_df[pc_col] == normalized_selection]

        agency_codes_applied = 0

        if agency_col and selected_agency_codes and len(selected_agency_codes) > 0:
            normalized_selections = [self._normalize_categorical_value(c) for c in selected_agency_codes]
            working_df = working_df[working_df[agency_col].isin(normalized_selections)]
            agency_codes_applied = len(normalized_selections)

        working_df = self._assign_business_type(working_df, id_col, time_col, biz_type_col)

        nb_df = working_df[working_df["BusinessType"] == "New"]
        ren_df = working_df[working_df["BusinessType"] == "Renewal"]

        business_split = {
            "new_business_premium": float(nb_df[fin_col].sum()) if not nb_df.empty else 0.0,
            "renewal_premium": float(ren_df[fin_col].sum()) if not ren_df.empty else 0.0,
            "new_business_count": int(nb_df[id_col].nunique()) if id_col and not nb_df.empty else int(len(nb_df)),
            "renewal_count": int(ren_df[id_col].nunique()) if id_col and not ren_df.empty else int(len(ren_df)),
            "classification_method": "explicit" if biz_type_col else ("derived" if id_col else "none")
        }

        if business_view == "new":
            working_df = working_df[working_df["BusinessType"] == "New"]
        elif business_view == "renewal":
            working_df = working_df[working_df["BusinessType"] == "Renewal"]

        effective_goals = goals if goals else (
            [{
                "id": "primary",
                "label": "Overall Goal",
                "period": "annual",
                "scope_type": "overall",
                "scope_value": None,
                "target_value": goal_value
            }] if goal_value and goal_value > 0 else []
        )

        if working_df.empty:
            empty_goal_progress = self.compute_goals(
                working_df, fin_col, id_col, time_col, pc_col, agency_col, cat_col,
                projection_target, effective_goals, effective_end
            )

            return {
                "kpis": {
                    "total_premium": 0,
                    "total_accounts": 0,
                    "avg_account_size": 0,
                    "retention_rate": 0,
                    "hhi_index": 0,
                    "pareto_ratio": 0
                },
                "historical_timeline": {
                    "labels": [],
                    "values": [],
                    "rolling_avg": [],
                    "mom_growth": [],
                    "new_values": [],
                    "renewal_values": []
                },
                "segment_distribution": {},
                "seasonality": {},
                "projections": [],
                "forecast_outlook": self._empty_forecast_outlook(projection_target),
                "ai_insights": self._empty_ai_insights(),
                "anomalies": [],
                "trend_anomalies": [],
                "vintage_cohorts": {},
                "goal_progress": empty_goal_progress,
                "business_split": business_split,
                "diagnostics": {
                    "future_records_removed": future_records_removed,
                    "future_dollar_amount": future_dollar_amount,
                    "include_future_dates": include_future_dates,
                    "agency_codes_applied": agency_codes_applied,
                    "business_view": business_view
                }
            }

        total_premium = float(working_df[fin_col].sum())
        total_accounts = int(working_df[id_col].nunique()) if id_col else int(len(working_df))
        avg_account_size = float(total_premium / total_accounts) if total_accounts > 0 else 0.0

        hhi_index = 0.0

        if cat_col and total_premium > 0:
            shares = working_df.groupby(cat_col)[fin_col].sum()
            hhi_index = float(sum([(v / total_premium * 100) ** 2 for v in shares]))

        working_df["Year"] = working_df[time_col].dt.year
        retention_rate = 100.0
        years_present = sorted(working_df["Year"].unique())

        if len(years_present) >= 2 and id_col:
            prev_year_accounts = set(working_df[working_df["Year"] == years_present[-2]][id_col].unique())
            curr_year_accounts = set(working_df[working_df["Year"] == years_present[-1]][id_col].unique())

            if prev_year_accounts:
                retained = prev_year_accounts.intersection(curr_year_accounts)
                retention_rate = float(len(retained) / len(prev_year_accounts) * 100)

        pareto_ratio = 20.0

        if id_col and total_premium > 0:
            account_sums = working_df.groupby(id_col)[fin_col].sum().sort_values(ascending=False)
            cumulative_sum = account_sums.cumsum()
            cutoff = total_premium * 0.80
            top_accounts_count = len(cumulative_sum[cumulative_sum <= cutoff]) + 1
            pareto_ratio = float((top_accounts_count / len(account_sums)) * 100) if len(account_sums) > 0 else 20.0

        vintage_cohorts = {}

        if id_col:
            first_seen = self.df.copy()
            first_seen[time_col] = pd.to_datetime(first_seen[time_col], errors="coerce")
            first_seen = first_seen.dropna(subset=[time_col])
            account_birthdays = first_seen.groupby(id_col)[time_col].min().dt.year.to_dict()
            working_df["Vintage"] = working_df[id_col].map(account_birthdays)

            vintage_metric_col = fin_col if projection_target == "premium" else id_col
            vintage_agg = "sum" if projection_target == "premium" else "nunique"

            vintage_summary = (
                working_df
                .groupby(["Vintage", "Year"])[vintage_metric_col]
                .agg(vintage_agg)
                .reset_index()
            )

            for v in vintage_summary["Vintage"].dropna().unique():
                v_str = f"Vintage {int(v)}"
                v_data = vintage_summary[vintage_summary["Vintage"] == v]

                vintage_cohorts[v_str] = {
                    f"CY_{int(row['Year'])}": float(row[vintage_metric_col])
                    for _, row in v_data.iterrows()
                }

        working_df["YearMonth"] = working_df[time_col].dt.to_period("M")
        monthly_groups = working_df.groupby("YearMonth")

        monthly_df = pd.DataFrame({
            "premium": monthly_groups[fin_col].sum(),
            "count": monthly_groups[id_col].nunique() if id_col else monthly_groups.size()
        }).reset_index()

        monthly_df["YearMonthStr"] = monthly_df["YearMonth"].astype(str)

        target_series = "premium" if projection_target == "premium" else "count"

        monthly_df["RollingAvg"] = monthly_df[target_series].rolling(window=3, min_periods=1).mean()
        monthly_df["MoM_Growth"] = (
            monthly_df[target_series]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0) * 100
        )

        nb_monthly = working_df[working_df["BusinessType"] == "New"].groupby("YearMonth")
        ren_monthly = working_df[working_df["BusinessType"] == "Renewal"].groupby("YearMonth")

        if projection_target == "premium":
            nb_series = nb_monthly[fin_col].sum()
            ren_series = ren_monthly[fin_col].sum()
        else:
            nb_series = nb_monthly[id_col].nunique() if id_col else nb_monthly.size()
            ren_series = ren_monthly[id_col].nunique() if id_col else ren_monthly.size()

        new_values = []
        renewal_values = []

        for ym in monthly_df["YearMonth"]:
            new_values.append(float(nb_series.get(ym, 0)))
            renewal_values.append(float(ren_series.get(ym, 0)))

        segment_data = {}

        if cat_col:
            segment_metric_col = fin_col if projection_target == "premium" else (id_col if id_col else fin_col)
            segment_agg = "sum" if projection_target == "premium" else "nunique"

            seg_summary = (
                working_df
                .groupby(cat_col)[segment_metric_col]
                .agg(segment_agg)
                .sort_values(ascending=False)
                .head(20)
            )

            segment_data = {str(k): float(v) for k, v in seg_summary.items()}

        working_df["MonthName"] = working_df[time_col].dt.strftime("%B")
        season_metric_col = fin_col if projection_target == "premium" else (id_col if id_col else fin_col)
        season_agg = "sum" if projection_target == "premium" else "nunique"

        season_summary = working_df.groupby("MonthName")[season_metric_col].agg(season_agg)
        seasonality = {k: float(v) for k, v in season_summary.to_dict().items()}

        projections = []

        if len(monthly_df) > 1:
            X = np.arange(len(monthly_df)).reshape(-1, 1)
            y = monthly_df[target_series].values

            model = LinearRegression().fit(X, y)
            future_X = np.arange(len(monthly_df), len(monthly_df) + 12).reshape(-1, 1)
            future_predictions = model.predict(future_X)

            last_date = working_df[time_col].max()

            for i, pred in enumerate(future_predictions):
                next_month = (last_date + pd.DateOffset(months=i + 1)).strftime("%Y-%m")

                projections.append({
                    "period": next_month,
                    "projected_value": max(0.0, float(pred))
                })

        forecast_outlook = self._compute_forecast_outlook(
            monthly_df=monthly_df,
            target_series=target_series,
            projection_target=projection_target
        )

        goal_progress = self.compute_goals(
            working_df, fin_col, id_col, time_col, pc_col, agency_col, cat_col,
            projection_target, effective_goals, effective_end
        )

        primary_goal = goal_progress[0] if goal_progress else None

        anomalies = []

        if id_col:
            top_accounts = working_df.groupby(id_col)[fin_col].sum().sort_values(ascending=False).head(10)

            for acc_id, acc_vol in top_accounts.items():
                if total_premium > 0 and (acc_vol / total_premium) > 0.03:
                    anomalies.append({
                        "identifier": str(acc_id),
                        "value": float(acc_vol),
                        "reason": (
                            f"High Concentration Exposure Outlier Risk "
                            f"({round(acc_vol / total_premium * 100, 1)}% of total selected scope)"
                        )
                    })

        trend_anomalies = []
        avg_monthly_scope_total = float(monthly_df[target_series].mean()) if not monthly_df.empty else 0.0
        trend_anomalies.extend(self._detect_trend_anomalies(
            working_df, time_col, fin_col, id_col, target_series, pc_col, "profit_center", avg_monthly_scope_total
        ))
        trend_anomalies.extend(self._detect_trend_anomalies(
            working_df, time_col, fin_col, id_col, target_series, agency_col, "agency_code", avg_monthly_scope_total
        ))
        trend_anomalies = sorted(trend_anomalies, key=lambda a: abs(a["deviation_pct"]), reverse=True)[:10]

        ai_insights = self._generate_ai_insights(
            total_premium=total_premium,
            total_accounts=total_accounts,
            avg_account_size=avg_account_size,
            retention_rate=retention_rate,
            hhi_index=hhi_index,
            pareto_ratio=pareto_ratio,
            business_split=business_split,
            forecast_outlook=forecast_outlook,
            primary_goal=primary_goal,
            segment_data=segment_data,
            anomalies=anomalies,
            projection_target=projection_target
        )

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
                "labels": monthly_df["YearMonthStr"].tolist(),
                "values": monthly_df[target_series].map(float).tolist(),
                "rolling_avg": monthly_df["RollingAvg"].map(float).tolist(),
                "mom_growth": monthly_df["MoM_Growth"].map(float).tolist(),
                "new_values": new_values,
                "renewal_values": renewal_values
            },
            "segment_distribution": segment_data,
            "seasonality": seasonality,
            "projections": projections,
            "forecast_outlook": forecast_outlook,
            "ai_insights": ai_insights,
            "anomalies": anomalies,
            "trend_anomalies": trend_anomalies,
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
    def _normalize_categorical_value(self, x):
        try:
            if isinstance(x, (float, int)) and not pd.isna(x) and x == int(x):
                return str(int(x))
        except (ValueError, TypeError):
            pass

        return str(x).strip()

    _normalize_pc_value = _normalize_categorical_value

    def _classify_business_type(self, value: str) -> str:
        if value is None:
            return "Other"

        v = str(value).strip().lower()

        if not v or v == "nan":
            return "Other"

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
        cleaned = [v for v in cleaned if v and v.lower() != "nan"]

        return sorted(list(set(cleaned)))[:limit]

    def get_profit_centers(self, pc_col: str) -> list:
        return self.get_unique_column_values(pc_col)

    def get_agency_codes(self, agency_col: str) -> list:
        return self.get_unique_column_values(agency_col)

    def get_date_range(self, time_col: str) -> dict:
        if not time_col or time_col not in self.df.columns:
            return {"min_date": None, "max_date": None}

        parsed = pd.to_datetime(self.df[time_col], errors="coerce").dropna()
        parsed = parsed[parsed >= ANALYTICAL_BASELINE]

        if parsed.empty:
            return {"min_date": None, "max_date": None}

        return {
            "min_date": parsed.min().strftime("%Y-%m-%d"),
            "max_date": parsed.max().strftime("%Y-%m-%d")
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

            if not schema["financial_metric"] and any(
                kw in col_lower for kw in ["premium", "revenue", "sales", "amount", "volume", "gwp"]
            ):
                schema["financial_metric"] = col
                continue

            if not schema["timeline_metric"] and any(
                kw in col_lower for kw in ["date", "effective", "renewal date", "inception"]
            ):
                schema["timeline_metric"] = col
                continue

            if not schema["business_type"] and any(
                kw in col_lower for kw in [
                    "business type", "biz type", "new/renewal", "nb/ren",
                    "transaction type", "policy type"
                ]
            ):
                schema["business_type"] = col
                continue

            if not schema["agency_code"] and any(
                kw in col_lower for kw in [
                    "agency", "agent code", "agency code", "producer code", "office code"
                ]
            ):
                schema["agency_code"] = col
                continue

            if not schema["profit_center"] and any(
                kw in col_lower for kw in ["profit", "center", "pc", "department", "unit", "branch"]
            ):
                schema["profit_center"] = col
                continue

            if not schema["categorical_segment"] and any(
                kw in col_lower for kw in ["broker", "region", "product", "lob", "type", "segment", "state"]
            ):
                schema["categorical_segment"] = col
                continue

            if not schema["client_id"] and any(
                kw in col_lower for kw in ["id", "client", "customer", "account", "name", "policy"]
            ):
                schema["client_id"] = col

        if not schema["financial_metric"]:
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                schema["financial_metric"] = num_cols[0]

        if not schema["timeline_metric"]:
            date_cols = self.df.select_dtypes(include=["datetime", "object"]).columns
            if len(date_cols) > 0:
                schema["timeline_metric"] = date_cols[0]

        if not schema["profit_center"]:
            cat_cols = self.df.select_dtypes(include=["object"]).columns
            if len(cat_cols) > 1:
                schema["profit_center"] = cat_cols[1]

        if not schema["categorical_segment"]:
            cat_cols = self.df.select_dtypes(include=["object"]).columns
            if len(cat_cols) > 0:
                schema["categorical_segment"] = cat_cols[0]

        return {
            "columns": columns,
            "inferred_mapping": schema,
            "profit_centers": self.get_profit_centers(schema["profit_center"])[:100],
            "agency_codes": self.get_agency_codes(schema["agency_code"])[:500],
            "date_range": self.get_date_range(schema["timeline_metric"]),
            "baseline_date": ANALYTICAL_BASELINE.strftime("%Y-%m-%d"),
            "has_business_type": schema["business_type"] is not None
        }

    def _empty_forecast_outlook(self, projection_target="premium"):
        return {
            "metric_type": projection_target,
            "current_year": None,
            "previous_year": None,
            "current_actual": 0.0,
            "previous_year_actual": 0.0,
            "projected_year_end": 0.0,
            "conservative_year_end": 0.0,
            "aggressive_year_end": 0.0,
            "remaining_months": 0,
            "growth_vs_previous_year_pct": 0.0,
            "confidence_score": 0.0,
            "confidence_label": "Insufficient Data",
            "trend_direction": "Flat",
            "monthly_forecast": [],
            "seasonal_index": {},
            "executive_summary": "Not enough data is available to produce a reliable year-end projection."
        }

    def _build_monthly_series(self, df, time_col, fin_col, id_col, target_series="premium"):
        """Aggregate a (already scoped/filtered) dataframe into a monthly time series."""
        if df is None or df.empty or time_col not in df.columns:
            return pd.DataFrame(columns=["YearMonth", "premium", "count"])

        d = df.copy()
        d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
        d = d.dropna(subset=[time_col])

        if d.empty:
            return pd.DataFrame(columns=["YearMonth", "premium", "count"])

        d["YearMonth"] = d[time_col].dt.to_period("M")
        grouped = d.groupby("YearMonth")

        monthly = pd.DataFrame({
            "premium": grouped[fin_col].sum() if fin_col in d.columns else grouped.size(),
            "count": grouped[id_col].nunique() if id_col and id_col in d.columns else grouped.size()
        }).reset_index()

        return monthly.sort_values("YearMonth").reset_index(drop=True)

    def _seasonal_trend_forecast(self, monthly_df, target_series, periods_ahead):
        """
        Produce a trend + seasonality aware forecast for the next `periods_ahead` months.

        Method: fit a linear trend across all available months, derive a per-calendar-month
        seasonal multiplier from the average ratio of actual-to-trend, normalize the
        multipliers to average 1.0, then project future months as trend * seasonal index.
        Confidence bands are built from the standard deviation of de-seasonalized residuals.
        """
        empty_diagnostics = {"r_squared": 0.0, "residual_std": 0.0, "slope": 0.0, "seasonal_index": {}}

        if monthly_df is None or monthly_df.empty or len(monthly_df) < 2 or periods_ahead <= 0:
            return [], empty_diagnostics

        mdf = monthly_df.copy().sort_values("YearMonth").reset_index(drop=True)
        mdf["CalMonth"] = mdf["YearMonth"].dt.month

        X = np.arange(len(mdf)).reshape(-1, 1)
        y = mdf[target_series].astype(float).values

        model = LinearRegression().fit(X, y)
        trend_fitted = model.predict(X)

        try:
            r_squared = float(model.score(X, y))
        except Exception:
            r_squared = 0.0

        safe_trend = np.where(np.abs(trend_fitted) < 1e-9, np.nan, trend_fitted)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = y / safe_trend

        mdf["_ratio"] = ratio

        seasonal_index = {}
        for m in range(1, 13):
            vals = mdf.loc[mdf["CalMonth"] == m, "_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
            seasonal_index[m] = float(vals.mean()) if len(vals) > 0 else 1.0

        observed_vals = [v for v in seasonal_index.values() if v is not None and not np.isnan(v)]
        mean_idx = float(np.mean(observed_vals)) if observed_vals else 1.0

        if mean_idx and not np.isnan(mean_idx) and mean_idx != 0:
            seasonal_index = {m: v / mean_idx for m, v in seasonal_index.items()}

        # Clip extreme seasonal swings so sparse-history months don't produce wild forecasts.
        seasonal_index = {m: float(np.clip(v, 0.4, 2.5)) for m, v in seasonal_index.items()}

        deseasonalized_fitted = trend_fitted * np.array([seasonal_index.get(m, 1.0) for m in mdf["CalMonth"]])
        residuals = y - deseasonalized_fitted
        residual_std = float(np.std(residuals)) if len(residuals) > 1 else 0.0

        last_period = mdf["YearMonth"].max()
        future_X = np.arange(len(mdf), len(mdf) + periods_ahead).reshape(-1, 1)
        future_trend = model.predict(future_X)

        future = []

        for i, base_pred in enumerate(future_trend):
            future_period = last_period + i + 1
            cal_month = int(future_period.month)
            seasonal_mult = seasonal_index.get(cal_month, 1.0)
            expected = max(0.0, float(base_pred) * seasonal_mult)
            conservative = max(0.0, expected - residual_std)
            aggressive = max(0.0, expected + residual_std)

            future.append({
                "period": str(future_period),
                "year": int(future_period.year),
                "month": cal_month,
                "expected_value": expected,
                "conservative_value": conservative,
                "aggressive_value": aggressive
            })

        slope = float(model.coef_[0]) if hasattr(model, "coef_") and len(model.coef_) > 0 else 0.0

        return future, {
            "r_squared": r_squared,
            "residual_std": residual_std,
            "slope": slope,
            "seasonal_index": seasonal_index
        }

    def _empty_ai_insights(self):
        return {
            "portfolio_health_score": 0.0,
            "portfolio_health_label": "Insufficient Data",
            "portfolio_health_status": "neutral",
            "executive_summary": "Not enough data is available to generate AI insights.",
            "insights": [],
            "recommended_actions": [
                "Upload or select a broader dataset to generate portfolio insights."
            ]
        }

    def _period_bounds(self, period: str, anchor: pd.Timestamp):
        """Return (start, end) timestamps for the annual/quarterly/monthly window containing `anchor`."""
        anchor = pd.Timestamp(anchor)

        if period == "monthly":
            start = pd.Timestamp(year=anchor.year, month=anchor.month, day=1)
            end = start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
        elif period == "quarterly":
            q_start_month = ((anchor.month - 1) // 3) * 3 + 1
            start = pd.Timestamp(year=anchor.year, month=q_start_month, day=1)
            end = start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        else:
            start = pd.Timestamp(year=anchor.year, month=1, day=1)
            end = pd.Timestamp(year=anchor.year, month=12, day=31)

        return start, end

    def _apply_goal_scope(self, df, scope_type, scope_value, pc_col, agency_col, cat_col):
        """Filter a dataframe down to the dimension a single goal targets."""
        if scope_type == "profit_center" and pc_col and pc_col in df.columns:
            normalized = self._normalize_categorical_value(scope_value)
            return df[df[pc_col] == normalized]

        if scope_type == "agency_code" and agency_col and agency_col in df.columns:
            normalized = self._normalize_categorical_value(scope_value)
            return df[df[agency_col] == normalized]

        if scope_type == "segment" and cat_col and cat_col in df.columns:
            return df[df[cat_col].astype(str) == str(scope_value)]

        return df

    def compute_goals(
        self,
        working_df,
        fin_col,
        id_col,
        time_col,
        pc_col,
        agency_col,
        cat_col,
        projection_target,
        goals,
        anchor_date
    ) -> list:
        """
        Compute pacing + a seasonality-aware projection for each goal in `goals`.

        Each goal dict supports: id, label, period ("annual"/"quarterly"/"monthly"),
        scope_type ("overall"/"profit_center"/"agency_code"/"segment"), scope_value, target_value.
        """
        target_series = "premium" if projection_target == "premium" else "count"
        results = []

        if not goals:
            return results

        if anchor_date is None or pd.isna(anchor_date):
            anchor_date = pd.Timestamp(datetime.now().date())

        for idx, g in enumerate(goals):
            goal_id = g.get("id") or f"goal_{idx + 1}"
            label = g.get("label") or "Goal"
            period = g.get("period") if g.get("period") in ("annual", "quarterly", "monthly") else "annual"
            scope_type = g.get("scope_type") if g.get("scope_type") in (
                "overall", "profit_center", "agency_code", "segment"
            ) else "overall"
            scope_value = g.get("scope_value")

            try:
                target_value = float(g.get("target_value") or 0)
            except (ValueError, TypeError):
                target_value = 0.0

            start, end = self._period_bounds(period, anchor_date)
            base_result = {
                "id": goal_id,
                "label": label,
                "period": period,
                "scope_type": scope_type,
                "scope_value": scope_value,
                "metric_type": projection_target,
                "period_start": start.strftime("%Y-%m-%d"),
                "period_end": end.strftime("%Y-%m-%d"),
                "days_total": int((end - start).days + 1)
            }

            scoped_df = self._apply_goal_scope(working_df, scope_type, scope_value, pc_col, agency_col, cat_col)

            if target_value <= 0:
                results.append({
                    **base_result,
                    "target": 0.0,
                    "actual": 0.0,
                    "achievement_pct": 0.0,
                    "expected_pct": 0.0,
                    "gap_to_goal": 0.0,
                    "projected_period_end": 0.0,
                    "days_elapsed": 0,
                    "status": "no_goal_set"
                })
                continue

            effective_end_for_actual = min(end, anchor_date)
            in_period = (scoped_df[time_col] >= start) & (scoped_df[time_col] <= effective_end_for_actual)
            period_df = scoped_df[in_period]

            if projection_target == "premium":
                actual = float(period_df[fin_col].sum()) if not period_df.empty else 0.0
            else:
                actual = float(period_df[id_col].nunique()) if id_col and not period_df.empty else float(len(period_df))

            days_total = max(1, base_result["days_total"])
            days_elapsed = max(0, (effective_end_for_actual - start).days + 1) if effective_end_for_actual >= start else 0

            expected_pct = min(100.0, (days_elapsed / days_total) * 100)
            achievement_pct = (actual / target_value) * 100
            gap_to_goal = float(actual - target_value)

            projected_period_end = float((actual / days_elapsed) * days_total) if days_elapsed > 0 else 0.0

            # Where at least one full calendar month remains in the goal period, refine the
            # run-rate projection using the seasonality-aware monthly forecast instead.
            anchor_month_period = pd.Period(anchor_date, freq="M")
            end_month_period = pd.Period(end, freq="M")

            if anchor_month_period < end_month_period:
                remaining_months = int(end_month_period.ordinal - anchor_month_period.ordinal)
                monthly_series = self._build_monthly_series(scoped_df, time_col, fin_col, id_col, target_series)

                if len(monthly_series) >= 2:
                    future_vals, _diag = self._seasonal_trend_forecast(monthly_series, target_series, remaining_months)
                    remaining_forecast = sum(item["expected_value"] for item in future_vals)
                    projected_period_end = float(actual + remaining_forecast)

            if expected_pct <= 0:
                status = "no_time_elapsed"
            else:
                pace_ratio = achievement_pct / expected_pct

                if pace_ratio >= 1.0:
                    status = "ahead"
                elif pace_ratio >= 0.8:
                    status = "on_pace"
                else:
                    status = "behind"

            results.append({
                **base_result,
                "target": float(target_value),
                "actual": float(actual),
                "achievement_pct": float(achievement_pct),
                "expected_pct": float(expected_pct),
                "gap_to_goal": float(gap_to_goal),
                "projected_period_end": float(projected_period_end),
                "days_elapsed": int(days_elapsed),
                "status": status
            })

        return results

    # ----------------------------------------------------------------------- #
    # Goal suggestions ("based on past info, here's what to aim for")
    # ----------------------------------------------------------------------- #

    def _reference_period_bounds(self, period: str, anchor: pd.Timestamp):
        """The same period one year prior — the historical baseline a suggested goal is built from."""
        start, end = self._period_bounds(period, anchor)
        return start - pd.DateOffset(years=1), end - pd.DateOffset(years=1)

    def _period_actual(self, df, time_col, fin_col, id_col, projection_target, start, end):
        if df is None or df.empty:
            return 0.0

        mask = (df[time_col] >= start) & (df[time_col] <= end)
        subset = df[mask]

        if subset.empty:
            return 0.0

        if projection_target == "premium":
            return float(subset[fin_col].sum())

        return float(subset[id_col].nunique()) if id_col else float(len(subset))

    def _estimate_growth_rate(self, df, time_col, fin_col, id_col, projection_target, period, anchor_date):
        """
        Compare the most recent complete comparable period to the one a year before it
        (e.g. last calendar year vs the year before). Returns None if there isn't enough
        history for this scope to compute a meaningful rate.
        """
        ref_start, ref_end = self._reference_period_bounds(period, anchor_date)
        prior_start, prior_end = ref_start - pd.DateOffset(years=1), ref_end - pd.DateOffset(years=1)

        ref_actual = self._period_actual(df, time_col, fin_col, id_col, projection_target, ref_start, ref_end)
        prior_actual = self._period_actual(df, time_col, fin_col, id_col, projection_target, prior_start, prior_end)

        if prior_actual <= 0:
            return None

        growth = (ref_actual - prior_actual) / prior_actual

        # Clip so a single noisy small-scope period can't produce an absurd suggestion.
        return float(np.clip(growth, -0.6, 1.5))

    def _round_target(self, value: float) -> float:
        """Round a suggested target to a clean, presentable figure (e.g. 6,432,911 -> 6,400,000)."""
        if value is None or value <= 0:
            return 0.0

        digits = len(str(int(value)))
        magnitude = max(1, 10 ** max(0, digits - 2))

        return float(round(value / magnitude) * magnitude)

    def suggest_goal_candidates(
        self,
        mapping: dict,
        projection_target: str = "premium",
        period: str = "annual",
        business_view: str = "all",
        top_n: int = 3
    ) -> list:
        """
        Propose goal candidates from historical performance: the overall book plus the
        top N profit centers, agencies, and segments by volume. Each candidate reports last
        year's actual for the same period, the book's own trailing growth rate, and three
        target tiers (maintain / grow / stretch) derived from those two numbers.
        """
        fin_col = mapping.get("financial_metric")
        time_col = mapping.get("timeline_metric")
        cat_col = mapping.get("categorical_segment")
        pc_col = mapping.get("profit_center")
        id_col = mapping.get("client_id")
        agency_col = mapping.get("agency_code")
        biz_type_col = mapping.get("business_type")

        if not fin_col or not time_col or fin_col not in self.df.columns or time_col not in self.df.columns:
            raise ValueError("Financial and Timeline columns must be mapped and present to suggest goals.")

        cat_col = cat_col if cat_col in self.df.columns else None
        pc_col = pc_col if pc_col in self.df.columns else None
        id_col = id_col if id_col in self.df.columns else None
        agency_col = agency_col if agency_col in self.df.columns else None
        biz_type_col = biz_type_col if biz_type_col in self.df.columns else None

        cols = list(dict.fromkeys(
            [fin_col, time_col] + [c for c in [cat_col, pc_col, id_col, agency_col, biz_type_col] if c]
        ))

        working_df = self.df[cols].copy()
        working_df[time_col] = pd.to_datetime(working_df[time_col], errors="coerce")
        working_df[fin_col] = pd.to_numeric(working_df[fin_col], errors="coerce").fillna(0)
        working_df = working_df.dropna(subset=[time_col])
        working_df = working_df[working_df[time_col] >= ANALYTICAL_BASELINE]

        if pc_col:
            working_df[pc_col] = working_df[pc_col].apply(self._normalize_categorical_value)
        if agency_col:
            working_df[agency_col] = working_df[agency_col].apply(self._normalize_categorical_value)
        if id_col:
            working_df[id_col] = working_df[id_col].fillna("Unknown")

        if working_df.empty:
            return []

        anchor_date = working_df[time_col].max()

        working_df = self._assign_business_type(working_df, id_col, time_col, biz_type_col)

        if business_view == "new":
            working_df = working_df[working_df["BusinessType"] == "New"]
        elif business_view == "renewal":
            working_df = working_df[working_df["BusinessType"] == "Renewal"]

        if working_df.empty:
            return []

        # Organization-wide growth rate, used as a fallback for scopes with too little
        # standalone history (e.g. a small agency with only a few months of data).
        overall_growth = self._estimate_growth_rate(
            working_df, time_col, fin_col, id_col, projection_target, period, anchor_date
        )

        scope_candidates = [("overall", None, "Overall Book")]

        def top_values(col):
            if not col or col not in working_df.columns:
                return []

            sums = working_df.groupby(col)[fin_col].sum().sort_values(ascending=False)
            return list(sums.head(top_n).index)

        for val in top_values(pc_col):
            scope_candidates.append(("profit_center", val, f"Profit Center: {val}"))
        for val in top_values(agency_col):
            scope_candidates.append(("agency_code", val, f"Agency: {val}"))
        for val in top_values(cat_col):
            scope_candidates.append(("segment", val, f"Segment: {val}"))

        suggestions = []

        for scope_type, scope_value, scope_label in scope_candidates:
            scoped_df = self._apply_goal_scope(working_df, scope_type, scope_value, pc_col, agency_col, cat_col)

            ref_start, ref_end = self._reference_period_bounds(period, anchor_date)
            cur_start, cur_end = self._period_bounds(period, anchor_date)

            reference_actual = self._period_actual(
                scoped_df, time_col, fin_col, id_col, projection_target, ref_start, ref_end
            )

            if reference_actual <= 0:
                continue

            growth = self._estimate_growth_rate(
                scoped_df, time_col, fin_col, id_col, projection_target, period, anchor_date
            )
            data_sufficient = growth is not None

            if growth is None:
                growth = overall_growth if overall_growth is not None else 0.0

            suggestions.append({
                "scope_type": scope_type,
                "scope_value": scope_value,
                "scope_label": scope_label,
                "period": period,
                "period_start": cur_start.strftime("%Y-%m-%d"),
                "period_end": cur_end.strftime("%Y-%m-%d"),
                "reference_period_start": ref_start.strftime("%Y-%m-%d"),
                "reference_period_end": ref_end.strftime("%Y-%m-%d"),
                "reference_actual": float(reference_actual),
                "growth_rate_pct": float(growth * 100),
                "data_sufficient": data_sufficient,
                "metric_type": projection_target,
                "suggestions": {
                    "maintain": self._round_target(reference_actual),
                    "grow": self._round_target(reference_actual * (1 + growth)),
                    "stretch": self._round_target(reference_actual * (1 + growth + 0.05))
                }
            })

        return suggestions

    def _assign_business_type(self, working_df, id_col, time_col, biz_type_col):
        if biz_type_col and biz_type_col in working_df.columns:
            working_df["BusinessType"] = working_df[biz_type_col].apply(self._classify_business_type)
            other_mask = working_df["BusinessType"] == "Other"

            if other_mask.any() and id_col and id_col in working_df.columns:
                first_seen = self.df.copy()
                first_seen[time_col] = pd.to_datetime(first_seen[time_col], errors="coerce")
                first_seen = first_seen.dropna(subset=[time_col])
                account_birthdays = first_seen.groupby(id_col)[time_col].min()

                def derive(row):
                    acc = row[id_col]

                    if acc in account_birthdays.index:
                        birthday = account_birthdays[acc]

                        if pd.notna(row[time_col]) and abs((row[time_col] - birthday).days) <= 30:
                            return "New"

                        return "Renewal"

                    return "Renewal"

                working_df.loc[other_mask, "BusinessType"] = working_df.loc[other_mask].apply(derive, axis=1)

        elif id_col and id_col in working_df.columns:
            first_seen = self.df.copy()
            first_seen[time_col] = pd.to_datetime(first_seen[time_col], errors="coerce")
            first_seen = first_seen.dropna(subset=[time_col])
            account_birthdays = first_seen.groupby(id_col)[time_col].min()

            def classify_row(row):
                acc = row[id_col]

                if acc in account_birthdays.index:
                    birthday = account_birthdays[acc]

                    if pd.notna(row[time_col]) and abs((row[time_col] - birthday).days) <= 30:
                        return "New"

                    return "Renewal"

                return "Renewal"

            working_df["BusinessType"] = working_df.apply(classify_row, axis=1)

        else:
            working_df["BusinessType"] = "Renewal"

        return working_df

    def _compute_forecast_outlook(self, monthly_df, target_series, projection_target):
        if monthly_df is None or monthly_df.empty or len(monthly_df) < 2:
            return self._empty_forecast_outlook(projection_target)

        forecast_df = monthly_df.copy().sort_values("YearMonth").reset_index(drop=True)
        forecast_df["Year"] = forecast_df["YearMonth"].dt.year

        current_year = int(forecast_df["Year"].max())
        previous_year = current_year - 1

        current_year_df = forecast_df[forecast_df["Year"] == current_year]
        previous_year_df = forecast_df[forecast_df["Year"] == previous_year]

        current_actual = float(current_year_df[target_series].sum())
        previous_year_actual = float(previous_year_df[target_series].sum()) if not previous_year_df.empty else 0.0

        last_period = forecast_df["YearMonth"].max()
        remaining_months = max(0, 12 - int(last_period.month))

        future_monthly, diagnostics = self._seasonal_trend_forecast(forecast_df, target_series, remaining_months)

        r_squared = diagnostics["r_squared"]
        residual_std = diagnostics["residual_std"]
        slope = diagnostics["slope"]

        avg_monthly = float(forecast_df[target_series].astype(float).mean()) if len(forecast_df) > 0 else 0.0
        volatility_ratio = abs(residual_std / avg_monthly) if avg_monthly != 0 else 1.0

        expected_future_total = sum(item["expected_value"] for item in future_monthly)
        conservative_future_total = sum(item["conservative_value"] for item in future_monthly)
        aggressive_future_total = sum(item["aggressive_value"] for item in future_monthly)

        projected_year_end = float(current_actual + expected_future_total)
        conservative_year_end = float(current_actual + conservative_future_total)
        aggressive_year_end = float(current_actual + aggressive_future_total)

        growth_vs_previous_year_pct = 0.0

        if previous_year_actual > 0:
            growth_vs_previous_year_pct = ((projected_year_end - previous_year_actual) / previous_year_actual) * 100

        if slope > 0:
            trend_direction = "Increasing"
        elif slope < 0:
            trend_direction = "Decreasing"
        else:
            trend_direction = "Flat"

        data_points = len(forecast_df)
        history_score = min(1.0, data_points / 12)
        fit_score = max(0.0, min(1.0, r_squared))
        stability_score = max(0.0, min(1.0, 1 - volatility_ratio))

        confidence_score = round(((history_score * 0.35) + (fit_score * 0.40) + (stability_score * 0.25)) * 100, 1)

        if data_points < 4:
            confidence_label = "Low"
        elif confidence_score >= 75:
            confidence_label = "High"
        elif confidence_score >= 50:
            confidence_label = "Moderate"
        else:
            confidence_label = "Low"

        metric_label = "premium volume" if projection_target == "premium" else "policy count"

        if previous_year_actual > 0:
            growth_phrase = f"{growth_vs_previous_year_pct:.1f}% compared with the prior year"
        else:
            growth_phrase = "no prior-year comparison is available"

        executive_summary = (
            f"Based on current monthly performance and seasonal patterns, the selected portfolio is projected to finish "
            f"{current_year} at approximately {projected_year_end:,.0f} in {metric_label}. "
            f"This represents {growth_phrase}. "
            f"The forecast confidence is {confidence_label.lower()} based on available history, trend fit, and volatility. "
            f"See the Goals panel for goal-specific pacing and projections."
        )

        return {
            "metric_type": projection_target,
            "current_year": current_year,
            "previous_year": previous_year,
            "current_actual": float(current_actual),
            "previous_year_actual": float(previous_year_actual),
            "projected_year_end": float(projected_year_end),
            "conservative_year_end": float(conservative_year_end),
            "aggressive_year_end": float(aggressive_year_end),
            "remaining_months": int(remaining_months),
            "growth_vs_previous_year_pct": float(growth_vs_previous_year_pct),
            "confidence_score": float(confidence_score),
            "confidence_label": confidence_label,
            "trend_direction": trend_direction,
            "monthly_forecast": future_monthly,
            "seasonal_index": diagnostics.get("seasonal_index", {}),
            "executive_summary": executive_summary
        }

    def _generate_ai_insights(
        self,
        total_premium,
        total_accounts,
        avg_account_size,
        retention_rate,
        hhi_index,
        pareto_ratio,
        business_split,
        forecast_outlook,
        primary_goal,
        segment_data,
        anomalies,
        projection_target
    ):
        insights = []
        action_items = []

        metric_label = "premium volume" if projection_target == "premium" else "policy count"

        projected_year_end = float(forecast_outlook.get("projected_year_end", 0) or 0)
        growth_pct = float(forecast_outlook.get("growth_vs_previous_year_pct", 0) or 0)
        confidence_label = forecast_outlook.get("confidence_label", "Insufficient Data")
        trend_direction = forecast_outlook.get("trend_direction", "Flat")

        goal_status = "No Goal Set"
        projected_gap_to_goal = 0.0

        if primary_goal and float(primary_goal.get("target", 0) or 0) > 0:
            projected_gap_to_goal = float(primary_goal.get("projected_period_end", 0) or 0) - float(primary_goal.get("target", 0) or 0)
            goal_status = "Projected Above Goal" if projected_gap_to_goal >= 0 else "Projected Below Goal"

        new_premium = float(business_split.get("new_business_premium", 0) or 0)
        renewal_premium = float(business_split.get("renewal_premium", 0) or 0)
        new_count = int(business_split.get("new_business_count", 0) or 0)
        renewal_count = int(business_split.get("renewal_count", 0) or 0)

        total_split_premium = new_premium + renewal_premium
        total_split_count = new_count + renewal_count

        new_premium_share = (new_premium / total_split_premium * 100) if total_split_premium > 0 else 0
        renewal_premium_share = (renewal_premium / total_split_premium * 100) if total_split_premium > 0 else 0
        new_count_share = (new_count / total_split_count * 100) if total_split_count > 0 else 0
        renewal_count_share = (renewal_count / total_split_count * 100) if total_split_count > 0 else 0

        top_segment_name = None
        top_segment_share = 0.0

        if segment_data and len(segment_data) > 0:
            try:
                top_segment_name = max(segment_data, key=segment_data.get)
                segment_total = sum(float(v or 0) for v in segment_data.values())

                if segment_total > 0:
                    top_segment_share = float(segment_data.get(top_segment_name, 0) or 0) / segment_total * 100
            except Exception:
                top_segment_name = None
                top_segment_share = 0.0

        anomaly_count = len(anomalies) if anomalies else 0

        if hhi_index >= 2500:
            concentration_level = "High"
        elif hhi_index >= 1500:
            concentration_level = "Moderate"
        else:
            concentration_level = "Low"

        if retention_rate >= 90:
            retention_level = "Strong"
        elif retention_rate >= 75:
            retention_level = "Watch"
        else:
            retention_level = "At Risk"

        if growth_pct >= 10:
            growth_level = "Accelerating"
        elif growth_pct >= 3:
            growth_level = "Growing"
        elif growth_pct <= -5:
            growth_level = "Declining"
        else:
            growth_level = "Flat"

        if projected_year_end > 0:
            insights.append({
                "category": "Forecast",
                "icon": "🔮",
                "severity": "positive" if growth_pct >= 0 else "warning",
                "title": "Year-End Projection",
                "message": (
                    f"Based on current monthly performance, the selected book is projected to finish at "
                    f"approximately {projected_year_end:,.0f} in {metric_label}. Forecast confidence is "
                    f"{confidence_label.lower()}."
                )
            })
        else:
            insights.append({
                "category": "Forecast",
                "icon": "🔮",
                "severity": "neutral",
                "title": "Forecast Availability",
                "message": "There is not enough monthly history available to produce a reliable year-end forecast."
            })

        if growth_level == "Accelerating":
            insights.append({
                "category": "Growth",
                "icon": "📈",
                "severity": "positive",
                "title": "Growth Momentum Is Strong",
                "message": (
                    f"The forecast indicates accelerating growth of approximately {growth_pct:.1f}% versus the prior year. "
                    f"Current trend direction is {trend_direction.lower()}."
                )
            })
            action_items.append(
                "Review the highest-performing segments to identify where growth is coming from and whether it can be replicated."
            )
        elif growth_level == "Growing":
            insights.append({
                "category": "Growth",
                "icon": "📈",
                "severity": "positive",
                "title": "Growth Trend Is Positive",
                "message": (
                    f"The book is projected to grow by approximately {growth_pct:.1f}% versus the prior year, "
                    f"suggesting positive but controlled expansion."
                )
            })
        elif growth_level == "Declining":
            insights.append({
                "category": "Growth",
                "icon": "📈",
                "severity": "risk",
                "title": "Growth Trend Is Declining",
                "message": (
                    f"The forecast indicates a decline of approximately {abs(growth_pct):.1f}% versus the prior year. "
                    f"This may require review of lost volume, renewal pressure, or reduced new business activity."
                )
            })
            action_items.append(
                "Investigate whether the decline is concentrated in specific profit centers, agencies, segments, or renewal groups."
            )
        else:
            insights.append({
                "category": "Growth",
                "icon": "📈",
                "severity": "neutral",
                "title": "Growth Trend Is Relatively Flat",
                "message": (
                    f"The current forecast shows limited movement versus the prior year at approximately {growth_pct:.1f}%."
                )
            })

        if goal_status == "Projected Above Goal":
            insights.append({
                "category": "Goal",
                "icon": "🏁",
                "severity": "positive",
                "title": "Projected Above Goal",
                "message": (
                    f"Current trends suggest the book may finish above goal by approximately "
                    f"{abs(projected_gap_to_goal):,.0f}."
                )
            })
        elif goal_status == "Projected Below Goal":
            insights.append({
                "category": "Goal",
                "icon": "🏁",
                "severity": "risk",
                "title": "Projected Below Goal",
                "message": (
                    f"Current trends suggest the book may finish below goal by approximately "
                    f"{abs(projected_gap_to_goal):,.0f}."
                )
            })
            action_items.append(
                "Compare the required monthly pace to recent monthly performance to determine whether the gap is realistically recoverable."
            )
        else:
            insights.append({
                "category": "Goal",
                "icon": "🏁",
                "severity": "neutral",
                "title": "No Goal Applied",
                "message": "No annual goal is currently applied, so goal-based variance is not being evaluated."
            })

        if retention_level == "Strong":
            insights.append({
                "category": "Retention",
                "icon": "✅",
                "severity": "positive",
                "title": "Retention Is Strong",
                "message": (
                    f"Account retention is currently {retention_rate:.1f}%, indicating healthy persistency across the selected book."
                )
            })
        elif retention_level == "Watch":
            insights.append({
                "category": "Retention",
                "icon": "✅",
                "severity": "warning",
                "title": "Retention Should Be Watched",
                "message": (
                    f"Account retention is currently {retention_rate:.1f}%. This is not critical, but it may deserve monitoring."
                )
            })
            action_items.append(
                "Look at renewal accounts by segment or agency to identify where retention is softening."
            )
        else:
            insights.append({
                "category": "Retention",
                "icon": "✅",
                "severity": "risk",
                "title": "Retention Risk Detected",
                "message": (
                    f"Account retention is currently {retention_rate:.1f}%, which may indicate elevated book persistence risk."
                )
            })
            action_items.append(
                "Prioritize reviewing accounts that were active in the prior year but are not appearing in the current year."
            )

        if concentration_level == "High":
            insights.append({
                "category": "Risk",
                "icon": "⚠️",
                "severity": "risk",
                "title": "High Concentration Risk",
                "message": (
                    f"The HHI concentration index is {hhi_index:,.0f}, which suggests elevated concentration exposure. "
                    f"{anomaly_count} concentration outlier account(s) were detected."
                )
            })
            action_items.append(
                "Review the largest accounts and determine whether the book is overly dependent on a small number of high-premium relationships."
            )
        elif concentration_level == "Moderate":
            insights.append({
                "category": "Risk",
                "icon": "⚠️",
                "severity": "warning",
                "title": "Moderate Concentration Risk",
                "message": (
                    f"The HHI concentration index is {hhi_index:,.0f}, suggesting moderate concentration. "
                    f"This is manageable, but still worth monitoring."
                )
            })
        else:
            insights.append({
                "category": "Risk",
                "icon": "⚠️",
                "severity": "positive",
                "title": "Concentration Appears Controlled",
                "message": (
                    f"The HHI concentration index is {hhi_index:,.0f}, suggesting the selected book is not overly concentrated."
                )
            })

        if total_split_premium > 0:
            if new_premium_share >= 35:
                insights.append({
                    "category": "Business Mix",
                    "icon": "🧭",
                    "severity": "positive",
                    "title": "New Business Contribution Is Strong",
                    "message": (
                        f"New business represents approximately {new_premium_share:.1f}% of premium volume, "
                        f"indicating strong contribution from new account activity."
                    )
                })
            elif new_premium_share >= 15:
                insights.append({
                    "category": "Business Mix",
                    "icon": "🧭",
                    "severity": "neutral",
                    "title": "Book Mix Is Renewal-Led With Meaningful New Business",
                    "message": (
                        f"Renewals represent approximately {renewal_premium_share:.1f}% of premium volume, "
                        f"while new business contributes {new_premium_share:.1f}%."
                    )
                })
            else:
                insights.append({
                    "category": "Business Mix",
                    "icon": "🧭",
                    "severity": "warning",
                    "title": "Book Is Heavily Renewal Dependent",
                    "message": (
                        f"New business represents only {new_premium_share:.1f}% of premium volume. "
                        f"The selected book appears highly dependent on renewals."
                    )
                })
                action_items.append(
                    "Review whether new business production is sufficient to offset future attrition risk."
                )
        elif total_split_count > 0:
            insights.append({
                "category": "Business Mix",
                "icon": "🧭",
                "severity": "neutral",
                "title": "Business Mix Available By Policy Count",
                "message": (
                    f"New business represents approximately {new_count_share:.1f}% of policies, while renewals represent "
                    f"{renewal_count_share:.1f}%."
                )
            })

        if top_segment_name:
            insights.append({
                "category": "Opportunity",
                "icon": "🎯",
                "severity": "positive" if top_segment_share >= 20 else "neutral",
                "title": "Largest Segment Opportunity",
                "message": (
                    f"{top_segment_name} is the largest visible segment in the selected scope, representing approximately "
                    f"{top_segment_share:.1f}% of measured volume. This segment may be useful for deeper opportunity review."
                )
            })

            if top_segment_share >= 35:
                action_items.append(
                    f"Evaluate whether {top_segment_name} concentration is strategic strength or a dependency risk."
                )

        if pareto_ratio <= 10:
            insights.append({
                "category": "Risk",
                "icon": "⚠️",
                "severity": "risk",
                "title": "Pareto Dependency Is Elevated",
                "message": (
                    f"Approximately {pareto_ratio:.1f}% of accounts appear to drive 80% of selected premium volume, "
                    f"which suggests a concentrated dependency profile."
                )
            })
        elif pareto_ratio <= 25:
            insights.append({
                "category": "Risk",
                "icon": "⚠️",
                "severity": "warning",
                "title": "Pareto Distribution Is Moderately Concentrated",
                "message": (
                    f"Approximately {pareto_ratio:.1f}% of accounts appear to drive 80% of selected premium volume."
                )
            })

        score = 100.0

        if growth_level == "Declining":
            score -= 20
        elif growth_level == "Flat":
            score -= 8

        if retention_level == "Watch":
            score -= 10
        elif retention_level == "At Risk":
            score -= 25

        if concentration_level == "Moderate":
            score -= 10
        elif concentration_level == "High":
            score -= 22

        if goal_status == "Projected Below Goal":
            score -= 15

        if confidence_label == "Low":
            score -= 8

        score = max(0.0, min(100.0, score))

        if score >= 85:
            health_label = "Excellent"
            health_status = "positive"
        elif score >= 70:
            health_label = "Healthy"
            health_status = "positive"
        elif score >= 55:
            health_label = "Watch"
            health_status = "warning"
        else:
            health_label = "At Risk"
            health_status = "risk"

        overview_parts = [
            f"The selected book is currently rated {health_label} with a portfolio health score of {score:.1f}/100."
        ]

        if projected_year_end > 0:
            overview_parts.append(
                f"The forecasted year-end position is approximately {projected_year_end:,.0f} in {metric_label}."
            )

        overview_parts.append(
            f"Retention is {retention_rate:.1f}% and concentration risk is classified as {concentration_level.lower()}."
        )

        if goal_status != "No Goal Set":
            overview_parts.append(f"Goal status is currently {goal_status.lower()}.")

        executive_summary = " ".join(overview_parts)

        if not action_items:
            action_items.append(
                "Continue monitoring forecast, retention, concentration, and new business mix as additional monthly data becomes available."
            )

        return {
            "portfolio_health_score": float(score),
            "portfolio_health_label": health_label,
            "portfolio_health_status": health_status,
            "executive_summary": executive_summary,
            "insights": insights,
            "recommended_actions": action_items[:5]
        }

    def run_analysis(
        self,
        mapping: dict,
        selected_profit_center: str = "ALL",
        projection_target: str = "premium",
        start_date: str = None,
        end_date: str = None,
        include_future_dates: bool = False,
        selected_agency_codes: list = None,
        goal_value: float = 0,
        goals: list = None,
        business_view: str = "all"
    ) -> dict:
        """
        Run the full book-of-business analysis pipeline: filter/scope the raw data per the
        request, compute KPIs, retention/concentration metrics, a seasonality-aware forecast,
        multi-goal pacing, and rule-based AI insights.

        Raises ValueError if the required financial/timeline columns are missing or not
        present in the uploaded file (e.g. a mapping left over from a previously loaded file).
        """
        if not isinstance(mapping, dict):
            raise ValueError("A column mapping is required to run analysis.")

        fin_col = mapping.get("financial_metric")
        time_col = mapping.get("timeline_metric")
        cat_col = mapping.get("categorical_segment")
        pc_col = mapping.get("profit_center")
        id_col = mapping.get("client_id")
        agency_col = mapping.get("agency_code")
        biz_type_col = mapping.get("business_type")

        if not fin_col or not time_col:
            raise ValueError("Financial and Timeline metrics are required fields.")

        if fin_col not in self.df.columns or time_col not in self.df.columns:
            raise ValueError(
                "The mapped Financial or Timeline column was not found in the uploaded file. "
                "This can happen if the mapping is left over from a different file — "
                "re-upload the file and confirm the schema mapping again."
            )

        # Optional mappings may reference a column that no longer exists (e.g. a stale mapping
        # or a saved view applied to a different file). Drop those silently rather than raising,
        # since the analysis can still run without them.
        cat_col = cat_col if cat_col in self.df.columns else None
        pc_col = pc_col if pc_col in self.df.columns else None
        id_col = id_col if id_col in self.df.columns else None
        agency_col = agency_col if agency_col in self.df.columns else None
        biz_type_col = biz_type_col if biz_type_col in self.df.columns else None

        cols_to_keep = [fin_col, time_col]

        if cat_col:
            cols_to_keep.append(cat_col)
        if pc_col:
            cols_to_keep.append(pc_col)
        if id_col:
            cols_to_keep.append(id_col)
        if agency_col:
            cols_to_keep.append(agency_col)
        if biz_type_col:
            cols_to_keep.append(biz_type_col)

        cols_to_keep = list(dict.fromkeys(cols_to_keep))

        working_df = self.df[cols_to_keep].copy()
        working_df[time_col] = pd.to_datetime(working_df[time_col], errors="coerce")
        working_df[fin_col] = pd.to_numeric(working_df[fin_col], errors="coerce").fillna(0)
        working_df = working_df.dropna(subset=[time_col])

        if pc_col:
            working_df[pc_col] = working_df[pc_col].apply(self._normalize_categorical_value)
        if agency_col:
            working_df[agency_col] = working_df[agency_col].apply(self._normalize_categorical_value)
        if cat_col:
            working_df[cat_col] = working_df[cat_col].fillna("Unknown")
        if id_col:
            working_df[id_col] = working_df[id_col].fillna("Unknown")

        working_df = working_df[working_df[time_col] >= ANALYTICAL_BASELINE]

        future_records_removed = 0
        future_dollar_amount = 0.0

        if not include_future_dates:
            today = pd.Timestamp(datetime.now().date())
            future_mask = working_df[time_col] > today
            future_records_removed = int(future_mask.sum())
            future_dollar_amount = float(working_df.loc[future_mask, fin_col].sum())
            working_df = working_df[~future_mask]

        effective_start = None
        effective_end = None

        if start_date:
            start_dt = pd.to_datetime(start_date, errors="coerce")

            if pd.notna(start_dt):
                working_df = working_df[working_df[time_col] >= start_dt]
                effective_start = start_dt

        if end_date:
            end_dt = pd.to_datetime(end_date, errors="coerce")

            if pd.notna(end_dt):
                end_capped = end_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)
                working_df = working_df[working_df[time_col] <= end_capped]
                effective_end = end_dt

        if effective_start is None and not working_df.empty:
            effective_start = working_df[time_col].min()

        if effective_end is None and not working_df.empty:
            effective_end = working_df[time_col].max()

        if pc_col and selected_profit_center and str(selected_profit_center).upper() != "ALL":
            normalized_selection = self._normalize_categorical_value(selected_profit_center)
            working_df = working_df[working_df[pc_col] == normalized_selection]

        agency_codes_applied = 0

        if agency_col and selected_agency_codes and len(selected_agency_codes) > 0:
            normalized_selections = [self._normalize_categorical_value(c) for c in selected_agency_codes]
            working_df = working_df[working_df[agency_col].isin(normalized_selections)]
            agency_codes_applied = len(normalized_selections)

        working_df = self._assign_business_type(working_df, id_col, time_col, biz_type_col)

        nb_df = working_df[working_df["BusinessType"] == "New"]
        ren_df = working_df[working_df["BusinessType"] == "Renewal"]

        business_split = {
            "new_business_premium": float(nb_df[fin_col].sum()) if not nb_df.empty else 0.0,
            "renewal_premium": float(ren_df[fin_col].sum()) if not ren_df.empty else 0.0,
            "new_business_count": int(nb_df[id_col].nunique()) if id_col and not nb_df.empty else int(len(nb_df)),
            "renewal_count": int(ren_df[id_col].nunique()) if id_col and not ren_df.empty else int(len(ren_df)),
            "classification_method": "explicit" if biz_type_col else ("derived" if id_col else "none")
        }

        if business_view == "new":
            working_df = working_df[working_df["BusinessType"] == "New"]
        elif business_view == "renewal":
            working_df = working_df[working_df["BusinessType"] == "Renewal"]

        effective_goals = goals if goals else (
            [{
                "id": "primary",
                "label": "Overall Goal",
                "period": "annual",
                "scope_type": "overall",
                "scope_value": None,
                "target_value": goal_value
            }] if goal_value and goal_value > 0 else []
        )

        if working_df.empty:
            empty_goal_progress = self.compute_goals(
                working_df, fin_col, id_col, time_col, pc_col, agency_col, cat_col,
                projection_target, effective_goals, effective_end
            )

            return {
                "kpis": {
                    "total_premium": 0,
                    "total_accounts": 0,
                    "avg_account_size": 0,
                    "retention_rate": 0,
                    "hhi_index": 0,
                    "pareto_ratio": 0
                },
                "historical_timeline": {
                    "labels": [],
                    "values": [],
                    "rolling_avg": [],
                    "mom_growth": [],
                    "new_values": [],
                    "renewal_values": []
                },
                "segment_distribution": {},
                "seasonality": {},
                "projections": [],
                "forecast_outlook": self._empty_forecast_outlook(projection_target),
                "ai_insights": self._empty_ai_insights(),
                "anomalies": [],
                "vintage_cohorts": {},
                "goal_progress": empty_goal_progress,
                "business_split": business_split,
                "diagnostics": {
                    "future_records_removed": future_records_removed,
                    "future_dollar_amount": future_dollar_amount,
                    "include_future_dates": include_future_dates,
                    "agency_codes_applied": agency_codes_applied,
                    "business_view": business_view
                }
            }

        total_premium = float(working_df[fin_col].sum())
        total_accounts = int(working_df[id_col].nunique()) if id_col else int(len(working_df))
        avg_account_size = float(total_premium / total_accounts) if total_accounts > 0 else 0.0

        hhi_index = 0.0

        if cat_col and total_premium > 0:
            shares = working_df.groupby(cat_col)[fin_col].sum()
            hhi_index = float(sum([(v / total_premium * 100) ** 2 for v in shares]))

        working_df["Year"] = working_df[time_col].dt.year
        retention_rate = 100.0
        years_present = sorted(working_df["Year"].unique())

        if len(years_present) >= 2 and id_col:
            prev_year_accounts = set(working_df[working_df["Year"] == years_present[-2]][id_col].unique())
            curr_year_accounts = set(working_df[working_df["Year"] == years_present[-1]][id_col].unique())

            if prev_year_accounts:
                retained = prev_year_accounts.intersection(curr_year_accounts)
                retention_rate = float(len(retained) / len(prev_year_accounts) * 100)

        pareto_ratio = 20.0

        if id_col and total_premium > 0:
            account_sums = working_df.groupby(id_col)[fin_col].sum().sort_values(ascending=False)
            cumulative_sum = account_sums.cumsum()
            cutoff = total_premium * 0.80
            top_accounts_count = len(cumulative_sum[cumulative_sum <= cutoff]) + 1
            pareto_ratio = float((top_accounts_count / len(account_sums)) * 100) if len(account_sums) > 0 else 20.0

        vintage_cohorts = {}

        if id_col:
            first_seen = self.df.copy()
            first_seen[time_col] = pd.to_datetime(first_seen[time_col], errors="coerce")
            first_seen = first_seen.dropna(subset=[time_col])
            account_birthdays = first_seen.groupby(id_col)[time_col].min().dt.year.to_dict()
            working_df["Vintage"] = working_df[id_col].map(account_birthdays)

            vintage_metric_col = fin_col if projection_target == "premium" else id_col
            vintage_agg = "sum" if projection_target == "premium" else "nunique"

            vintage_summary = (
                working_df
                .groupby(["Vintage", "Year"])[vintage_metric_col]
                .agg(vintage_agg)
                .reset_index()
            )

            for v in vintage_summary["Vintage"].dropna().unique():
                v_str = f"Vintage {int(v)}"
                v_data = vintage_summary[vintage_summary["Vintage"] == v]

                vintage_cohorts[v_str] = {
                    f"CY_{int(row['Year'])}": float(row[vintage_metric_col])
                    for _, row in v_data.iterrows()
                }

        working_df["YearMonth"] = working_df[time_col].dt.to_period("M")
        monthly_groups = working_df.groupby("YearMonth")

        monthly_df = pd.DataFrame({
            "premium": monthly_groups[fin_col].sum(),
            "count": monthly_groups[id_col].nunique() if id_col else monthly_groups.size()
        }).reset_index()

        monthly_df["YearMonthStr"] = monthly_df["YearMonth"].astype(str)

        target_series = "premium" if projection_target == "premium" else "count"

        monthly_df["RollingAvg"] = monthly_df[target_series].rolling(window=3, min_periods=1).mean()
        monthly_df["MoM_Growth"] = (
            monthly_df[target_series]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0) * 100
        )

        nb_monthly = working_df[working_df["BusinessType"] == "New"].groupby("YearMonth")
        ren_monthly = working_df[working_df["BusinessType"] == "Renewal"].groupby("YearMonth")

        if projection_target == "premium":
            nb_series = nb_monthly[fin_col].sum()
            ren_series = ren_monthly[fin_col].sum()
        else:
            nb_series = nb_monthly[id_col].nunique() if id_col else nb_monthly.size()
            ren_series = ren_monthly[id_col].nunique() if id_col else ren_monthly.size()

        new_values = []
        renewal_values = []

        for ym in monthly_df["YearMonth"]:
            new_values.append(float(nb_series.get(ym, 0)))
            renewal_values.append(float(ren_series.get(ym, 0)))

        segment_data = {}

        if cat_col:
            segment_metric_col = fin_col if projection_target == "premium" else (id_col if id_col else fin_col)
            segment_agg = "sum" if projection_target == "premium" else "nunique"

            seg_summary = (
                working_df
                .groupby(cat_col)[segment_metric_col]
                .agg(segment_agg)
                .sort_values(ascending=False)
                .head(20)
            )

            segment_data = {str(k): float(v) for k, v in seg_summary.items()}

        working_df["MonthName"] = working_df[time_col].dt.strftime("%B")
        season_metric_col = fin_col if projection_target == "premium" else (id_col if id_col else fin_col)
        season_agg = "sum" if projection_target == "premium" else "nunique"

        season_summary = working_df.groupby("MonthName")[season_metric_col].agg(season_agg)
        seasonality = {k: float(v) for k, v in season_summary.to_dict().items()}

        projections = []

        if len(monthly_df) > 1:
            X = np.arange(len(monthly_df)).reshape(-1, 1)
            y = monthly_df[target_series].values

            model = LinearRegression().fit(X, y)
            future_X = np.arange(len(monthly_df), len(monthly_df) + 12).reshape(-1, 1)
            future_predictions = model.predict(future_X)

            last_date = working_df[time_col].max()

            for i, pred in enumerate(future_predictions):
                next_month = (last_date + pd.DateOffset(months=i + 1)).strftime("%Y-%m")

                projections.append({
                    "period": next_month,
                    "projected_value": max(0.0, float(pred))
                })

        forecast_outlook = self._compute_forecast_outlook(
            monthly_df=monthly_df,
            target_series=target_series,
            projection_target=projection_target
        )

        goal_progress = self.compute_goals(
            working_df, fin_col, id_col, time_col, pc_col, agency_col, cat_col,
            projection_target, effective_goals, effective_end
        )

        primary_goal = goal_progress[0] if goal_progress else None

        anomalies = []

        if id_col:
            top_accounts = working_df.groupby(id_col)[fin_col].sum().sort_values(ascending=False).head(10)

            for acc_id, acc_vol in top_accounts.items():
                if total_premium > 0 and (acc_vol / total_premium) > 0.03:
                    anomalies.append({
                        "identifier": str(acc_id),
                        "value": float(acc_vol),
                        "reason": (
                            f"High Concentration Exposure Outlier Risk "
                            f"({round(acc_vol / total_premium * 100, 1)}% of total selected scope)"
                        )
                    })

        ai_insights = self._generate_ai_insights(
            total_premium=total_premium,
            total_accounts=total_accounts,
            avg_account_size=avg_account_size,
            retention_rate=retention_rate,
            hhi_index=hhi_index,
            pareto_ratio=pareto_ratio,
            business_split=business_split,
            forecast_outlook=forecast_outlook,
            primary_goal=primary_goal,
            segment_data=segment_data,
            anomalies=anomalies,
            projection_target=projection_target
        )

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
                "labels": monthly_df["YearMonthStr"].tolist(),
                "values": monthly_df[target_series].map(float).tolist(),
                "rolling_avg": monthly_df["RollingAvg"].map(float).tolist(),
                "mom_growth": monthly_df["MoM_Growth"].map(float).tolist(),
                "new_values": new_values,
                "renewal_values": renewal_values
            },
            "segment_distribution": segment_data,
            "seasonality": seasonality,
            "projections": projections,
            "forecast_outlook": forecast_outlook,
            "ai_insights": ai_insights,
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
