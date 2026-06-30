import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io

class BookOfBusinessAnalyzer:
    def __init__(self, file_bytes: bytes, file_name: str):
        self.file_name = file_name
        if file_name.endswith('.csv'):
            self.df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            self.df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel.")
        
        # Clean string columns to prevent trailing spaces and handle empty fields safely
        self.df = self.df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    def infer_schema(self) -> dict:
        """
        Heuristically identifies column roles based on names and data types.
        """
        schema = {
            "financial_metric": None,
            "timeline_metric": None,
            "categorical_segment": None,
            "profit_center": None,
            "client_id": None
        }
        
        columns = self.df.columns.tolist()
        
        for col in columns:
            col_lower = str(col).lower()
            
            # Financial Mapping
            if not schema["financial_metric"]:
                if any(kw in col_lower for kw in ["premium", "revenue", "sales", "amount", "volume", "gwp"]):
                    schema["financial_metric"] = col
                    continue
            
            # Timeline Mapping
            if not schema["timeline_metric"]:
                if any(kw in col_lower for kw in ["date", "effective", "renewal", "inception", "year"]):
                    schema["timeline_metric"] = col
                    continue
                    
            # Profit Center Mapping
            if not schema["profit_center"]:
                if any(kw in col_lower for kw in ["profit", "center", "pc", "department", "unit", "branch"]):
                    schema["profit_center"] = col
                    continue

            # Categorical Mapping
            if not schema["categorical_segment"]:
                if any(kw in col_lower for kw in ["broker", "region", "product", "lob", "type", "segment", "state"]):
                    schema["categorical_segment"] = col
                    continue

            # Fallback Client Identifier
            if not schema["client_id"]:
                if any(kw in col_lower for kw in ["id", "client", "customer", "account", "name"]):
                    schema["client_id"] = col

        # Absolute fallbacks if keywords miss
        if not schema["financial_metric"]:
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0: schema["financial_metric"] = num_cols[0]
            
        if not schema["timeline_metric"]:
            date_cols = self.df.select_dtypes(include=['datetime', 'object']).columns
            if len(date_cols) > 0: schema["timeline_metric"] = date_cols[0]

        if not schema["profit_center"]:
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 1: schema["profit_center"] = cat_cols[1]
        if not schema["categorical_segment"]:
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0: schema["categorical_segment"] = cat_cols[0]

        # Extract unique profit centers safely handling any internal NaN fields
        unique_profit_centers = []
        if schema["profit_center"] and schema["profit_center"] in self.df.columns:
            unique_profit_centers = sorted(
                self.df[schema["profit_center"]].dropna().astype(str).unique().tolist()
            )

        return {
            "columns": columns, 
            "inferred_mapping": schema,
            "profit_centers": unique_profit_centers
        }

    def run_analysis(self, mapping: dict, selected_profit_center: str = "ALL", projection_target: str = "premium") -> dict:
        """
        Executes core analytical computations filtered from year 2022 onwards.
        """
        fin_col = mapping.get("financial_metric")
        time_col = mapping.get("timeline_metric")
        cat_col = mapping.get("categorical_segment")
        pc_col = mapping.get("profit_center")
        id_col = mapping.get("client_id")

        if not fin_col or not time_col:
            raise ValueError("Financial and Timeline metrics are required for complete analysis.")

        # Data Normalization
        working_df = self.df.copy()
        working_df[time_col] = pd.to_datetime(working_df[time_col], errors='coerce')
        working_df[fin_col] = pd.to_numeric(working_df[fin_col], errors='coerce').fillna(0)
        working_df = working_df.dropna(subset=[time_col])

        # Fill potential structural gaps in critical categoricals to prevent grouping breakdowns
        if cat_col: working_df[cat_col] = working_df[cat_col].fillna('Unknown')
        if pc_col: working_df[pc_col] = working_df[pc_col].fillna('Unknown')
        if id_col: working_df[id_col] = working_df[id_col].fillna('Unknown')

        # CRITICAL FILTER: Restrict data analysis scope starting from 2022
        working_df = working_df[working_df[time_col].dt.year >= 2022]
        working_df = working_df.sort_values(by=time_col)

        # Dynamic Segment Filter: Slice by Profit Center if requested
        if pc_col and pc_col in working_df.columns and selected_profit_center != "ALL":
            working_df = working_df[working_df[pc_col].astype(str) == selected_profit_center]

        if working_df.empty:
            return {
                "kpis": {"total_premium": 0, "total_accounts": 0, "avg_account_size": 0},
                "historical_timeline": {"labels": [], "values": [], "rolling_avg": [], "mom_growth": []},
                "segment_distribution": {}, "seasonality": {}, "projections": [], "anomalies": []
            }

        # 1. High Level KPIs
        total_premium = float(working_df[fin_col].sum())
        total_accounts = int(working_df[id_col].nunique()) if id_col else int(len(working_df))
        avg_account_size = float(working_df[fin_col].mean()) if total_accounts > 0 else 0

        # 2. Historical Trends & Rolling Metrics
        working_df['YearMonth'] = working_df[time_col].dt.to_period('M')
        
        # Calculate monthly totals for premium vs policy count
        monthly_groups = working_df.groupby('YearMonth')
        monthly_premium = monthly_groups[fin_col].sum()
        monthly_count = monthly_groups[id_col].nunique() if id_col else monthly_groups.size()

        monthly_df = pd.DataFrame({
            'premium': monthly_premium,
            'count': monthly_count
        }).reset_index()
        monthly_df['YearMonthStr'] = monthly_df['YearMonth'].astype(str)
        
        # Choose targets based on interactive state configuration
        target_series = 'premium' if projection_target == "premium" else 'count'
        
        # Rolling Average (3-Month Window)
        monthly_df['RollingAvg'] = monthly_df[target_series].rolling(window=3, min_periods=1).mean()
        
        # Period-over-Period Growth
        monthly_df['MoM_Growth'] = monthly_df[target_series].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0) * 100

        # 3. Segment Performance
        segment_data = {}
        if cat_col and cat_col in working_df.columns:
            seg_summary = working_df.groupby(cat_col)[fin_col if projection_target == "premium" else (id_col if id_col else fin_col)].agg('sum' if projection_target == "premium" else 'nunique').sort_values(ascending=False)
            segment_data = {str(k): float(v) for k, v in seg_summary.items() if not np.isinf(v) and not np.isnan(v)}

        # 4. Root Cause Analysis / Seasonality Profile
        working_df['MonthName'] = working_df[time_col].dt.strftime('%B')
        season_summary = working_df.groupby('MonthName')[fin_col if projection_target == "premium" else (id_col if id_col else fin_col)].agg('mean' if projection_target == "premium" else 'nunique')
        seasonality = {k: (float(v) if not np.isinf(v) and not np.isnan(v) else 0.0) for k, v in season_summary.to_dict().items()}

        # 5. Predictive Time-Series Projections
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
                val = float(pred) if not np.isinf(pred) and not np.isnan(pred) else 0.0
                projections.append({"period": next_month, "projected_value": max(0.0, val)})

        # 6. Statistical Anomaly Detection
        anomalies = []
        if len(working_df) > 3:
            mean_val = working_df[fin_col].mean()
            std_val = working_df[fin_col].std()
            if std_val > 0:
                working_df['Z_Score'] = (working_df[fin_col] - mean_val) / std_val
                outliers = working_df[abs(working_df['Z_Score']) > 2.0]
                for idx, row in outliers.iterrows():
                    val = float(row[fin_col])
                    if not np.isinf(val) and not np.isnan(val):
                        anomalies.append({
                            "identifier": str(row[id_col]) if id_col else f"Row {idx}",
                            "value": val,
                            "reason": "Extreme deviation from baseline premium averages"
                        })

        clean_timeline_values = [float(v) if not np.isinf(v) and not np.isnan(v) else 0.0 for v in monthly_df[target_series].tolist()]
        clean_rolling_avg = [float(v) if not np.isinf(v) and not np.isnan(v) else 0.0 for v in monthly_df['RollingAvg'].tolist()]
        clean_mom_growth = [float(v) if not np.isinf(v) and not np.isnan(v) else 0.0 for v in monthly_df['MoM_Growth'].tolist()]

        return {
            "kpis": {
                "total_premium": total_premium,
                "total_accounts": total_accounts,
                "avg_account_size": avg_account_size
            },
            "historical_timeline": {
                "labels": monthly_df['YearMonthStr'].tolist(),
                "values": clean_timeline_values,
                "rolling_avg": clean_rolling_avg,
                "mom_growth": clean_mom_growth
            },
            "segment_distribution": segment_data,
            "seasonality": seasonality,
            "projections": projections,
            "anomalies": anomalies[:10]
        }
