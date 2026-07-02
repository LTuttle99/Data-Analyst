import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io

class BookOfBusinessAnalyzer:
    def __init__(self, file_bytes: bytes, file_name: str):
        self.file_name = file_name
        
        # Performance tuning: Read file streams directly without wrapping everything in memory structures
        if file_name.endswith('.csv'):
            self.df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
        elif file_name.endswith(('.xls', '.xlsx')):
            self.df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel.")
        
        # Memory optimization: Strip white spaces ONLY on object columns without copying the whole dataframe
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].astype(str).str.strip()

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
            
            if not schema["financial_metric"] and any(kw in col_lower for kw in ["premium", "revenue", "sales", "amount", "volume", "gwp"]):
                schema["financial_metric"] = col
                continue
            if not schema["timeline_metric"] and any(kw in col_lower for kw in ["date", "effective", "renewal", "inception", "year"]):
                schema["timeline_metric"] = col
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

        # RAM safety: Only extract top unique profit centers, dropping structural garbage or overflows
        unique_profit_centers = []
        if schema["profit_center"] and schema["profit_center"] in self.df.columns:
            unique_profit_centers = sorted(
                self.df[schema["profit_center"]].dropna().astype(str).unique().tolist()
            )
            # Cap array bounds to prevent HTML selective payload bloat
            unique_profit_centers = unique_profit_centers[:100]

        return {
            "columns": columns, 
            "inferred_mapping": schema,
            "profit_centers": unique_profit_centers
        }

    def run_analysis(self, mapping: dict, selected_profit_center: str = "ALL", projection_target: str = "premium") -> dict:
        """
        Executes highly optimized memory-bounded calculations filtered from 2022 onwards.
        """
        fin_col = mapping.get("financial_metric")
        time_col = mapping.get("timeline_metric")
        cat_col = mapping.get("categorical_segment")
        pc_col = mapping.get("profit_center")
        id_col = mapping.get("client_id")

        if not fin_col or not time_col:
            raise ValueError("Financial and Timeline metrics are required for complete analysis.")

        # Vectorized parsing limits footprint optimization
        working_df = self.df[[fin_col, time_col] + 
                             ([cat_col] if cat_col else []) + 
                             ([pc_col] if pc_col else []) + 
                             ([id_col] if id_col else [])].copy()

        working_df[time_col] = pd.to_datetime(working_df[time_col], errors='coerce')
        working_df[fin_col] = pd.to_numeric(working_df[fin_col], errors='coerce').fillna(0)
        working_df = working_df.dropna(subset=[time_col])

        # CRITICAL FILTER: Instantly prune the 5200 rows to 2022+ to drop dead weight from RAM
        working_df = working_df[working_df[time_col].dt.year >= 2022]

        if pc_col and selected_profit_center != "ALL":
            working_df = working_df[working_df[pc_col].astype(str) == selected_profit_center]

        if working_df.empty:
            return {
                "kpis": {"total_premium": 0, "total_accounts": 0, "avg_account_size": 0},
                "historical_timeline": {"labels": [], "values": [], "rolling_avg": [], "mom_growth": []},
                "segment_distribution": {}, "seasonality": {}, "projections": [], "anomalies": []
            }

        # 1. KPIs
        total_premium = float(working_df[fin_col].sum())
        total_accounts = int(working_df[id_col].nunique()) if id_col else int(len(working_df))
        avg_account_size = float(working_df[fin_col].mean()) if total_accounts > 0 else 0

        # 2. Historical Aggregations
        working_df['YearMonth'] = working_df[time_col].dt.to_period('M')
        monthly_groups = working_df.groupby('YearMonth')
        
        monthly_premium = monthly_groups[fin_col].sum()
        monthly_count = monthly_groups[id_col].nunique() if id_col else monthly_groups.size()

        monthly_df = pd.DataFrame({'premium': monthly_premium, 'count': monthly_count}).reset_index()
        monthly_df['YearMonthStr'] = monthly_df['YearMonth'].astype(str)
        
        target_series = 'premium' if projection_target == "premium" else 'count'
        monthly_df['RollingAvg'] = monthly_df[target_series].rolling(window=3, min_periods=1).mean()
        monthly_df['MoM_Growth'] = monthly_df[target_series].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0) * 100

        # 3. Segment Performance (Cap at top 15 blocks to prevent frontend rendering lockups)
        segment_data = {}
        if cat_col:
            seg_summary = working_df.groupby(cat_col)[fin_col if projection_target == "premium" else (id_col if id_col else fin_col)].agg('sum' if projection_target == "premium" else 'nunique').sort_values(ascending=False).head(15)
            segment_data = {str(k): float(v) for k, v in seg_summary.items()}

        # 4. Seasonality Profile
        working_df['MonthName'] = working_df[time_col].dt.strftime('%B')
        season_summary = working_df.groupby('MonthName')[fin_col if projection_target == "premium" else (id_col if id_col else fin_col)].agg('mean' if projection_target == "premium" else 'nunique')
        seasonality = {k: float(v) for k, v in season_summary.to_dict().items()}

        # 5. Projections
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

        # 6. Optimized Anomaly Detection (Avoid sorting massive tracking rows)
        anomalies = []
        if len(working_df) > 3:
            mean_val = working_df[fin_col].mean()
            std_val = working_df[fin_col].std()
            if std_val > 0:
                working_df['Z_Score'] = (working_df[fin_col] - mean_val) / std_val
                outliers = working_df[abs(working_df['Z_Score']) > 2.5].head(10) # 2.5 Sigma + Top 10 limit
                for idx, row in outliers.iterrows():
                    anomalies.append({
                        "identifier": str(row[id_col]) if id_col else f"Row {idx}",
                        "value": float(row[fin_col]),
                        "reason": "High variance signature outlier"
                    })

        return {
            "kpis": {"total_premium": total_premium, "total_accounts": total_accounts, "avg_account_size": avg_account_size},
            "historical_timeline": {
                "labels": monthly_df['YearMonthStr'].tolist(),
                "values": monthly_df[target_series].map(float).tolist(),
                "rolling_avg": monthly_df['RollingAvg'].map(float).tolist(),
                "mom_growth": monthly_df['MoM_Growth'].map(float).tolist()
            },
            "segment_distribution": segment_data,
            "seasonality": seasonality,
            "projections": projections,
            "anomalies": anomalies
        }
