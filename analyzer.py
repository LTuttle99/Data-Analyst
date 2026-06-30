import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io
import json

class BookOfBusinessAnalyzer:
    def __init__(self, file_bytes: bytes, file_name: str):
        self.file_name = file_name
        if file_name.endswith('.csv'):
            self.df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            self.df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel.")
        
        # Clean string columns to prevent trailing spaces
        self.df = self.df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    def infer_schema(self) -> dict:
        """
        Heuristically identifies column roles based on names and data types.
        """
        schema = {
            "financial_metric": None,
            "timeline_metric": None,
            "categorical_segment": None,
            "client_id": None
        }
        
        columns = self.df.columns.tolist()
        
        # Lowercase mapping for keywords
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

        if not schema["categorical_segment"]:
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0: schema["categorical_segment"] = cat_cols[0]

        return {"columns": columns, "inferred_mapping": schema}

    def run_analysis(self, mapping: dict) -> dict:
        """
        Executes core analytical computations using user-confirmed mappings.
        """
        fin_col = mapping.get("financial_metric")
        time_col = mapping.get("timeline_metric")
        cat_col = mapping.get("categorical_segment")
        id_col = mapping.get("client_id")

        if not fin_col or not time_col:
            raise ValueError("Financial and Timeline metrics are required for complete analysis.")

        # Data Normalization
        working_df = self.df.copy()
        working_df[time_col] = pd.to_datetime(working_df[time_col], errors='coerce')
        working_df[fin_col] = pd.to_numeric(working_df[fin_col], errors='coerce').fillna(0)
        working_df = working_df.dropna(subset=[time_col]).sort_values(by=time_col)

        # 1. High Level KPIs
        total_premium = float(working_df[fin_col].sum())
        total_accounts = int(working_df[id_col].nunique()) if id_col else int(len(working_df))
        avg_account_size = float(working_df[fin_col].mean()) if total_accounts > 0 else 0

        # 2. Historical Trends & Rolling Metrics
        working_df['YearMonth'] = working_df[time_col].dt.to_period('M')
        monthly_df = working_df.groupby('YearMonth')[fin_col].sum().reset_index()
        monthly_df['YearMonthStr'] = monthly_df['YearMonth'].astype(str)
        
        # Rolling Average (3-Month Window)
        monthly_df['RollingAvg'] = monthly_df[fin_col].rolling(window=3, min_periods=1).mean()
        
        # Period-over-Period Growth
        monthly_df['MoM_Growth'] = monthly_df[fin_col].pct_change().fillna(0) * 100

        # 3. Segment Performance
        segment_data = {}
        if cat_col and cat_col in working_df.columns:
            seg_summary = working_df.groupby(cat_col)[fin_col].sum().sort_values(ascending=False)
            segment_data = {str(k): float(v) for k, v in seg_summary.items()}

        # 4. Root Cause Analysis (Correlation of low revenue periods vs month/segments)
        working_df['MonthName'] = working_df[time_col].dt.strftime('%B')
        seasonality = working_df.groupby('MonthName')[fin_col].mean().to_dict()

        # 5. Predictive Time-Series Projections (Linear Regression Trendlines)
        projections = []
        if len(monthly_df) > 1:
            X = np.arange(len(monthly_df)).reshape(-1, 1)
            y = monthly_df[fin_col].values
            model = LinearRegression().fit(X, y)
            
            # Predict next 12 periods
            future_X = np.arange(len(monthly_df), len(monthly_df) + 12).reshape(-1, 1)
            future_predictions = model.predict(future_X)
            
            last_date = working_df[time_col].max()
            for i, pred in enumerate(future_predictions):
                next_month = (last_date + pd.DateOffset(months=i+1)).strftime('%Y-%m')
                projections.append({"period": next_month, "projected_value": max(0.0, float(pred))})

        # 6. Statistical Anomaly Detection (Z-Score > 2.0 or Outliers)
        anomalies = []
        if len(working_df) > 3:
            mean_val = working_df[fin_col].mean()
            std_val = working_df[fin_col].std()
            if std_val > 0:
                working_df['Z_Score'] = (working_df[fin_col] - mean_val) / std_val
                outliers = working_df[abs(working_df['Z_Score']) > 2.0]
                for idx, row in outliers.iterrows():
                    anomalies.append({
                        "identifier": str(row[id_col]) if id_col else f"Row {idx}",
                        "value": float(row[fin_col]),
                        "reason": "Extreme deviation from base metrics (High Z-Score)" if row['Z_Score'] > 0 else "Unusual volume dropped variant"
                    })

        return {
            "kpis": {
                "total_premium": total_premium,
                "total_accounts": total_accounts,
                "avg_account_size": avg_account_size
            },
            "historical_timeline": {
                "labels": monthly_df['YearMonthStr'].tolist(),
                "values": monthly_df[fin_col].tolist(),
                "rolling_avg": monthly_df['RollingAvg'].tolist(),
                "mom_growth": monthly_df['MoM_Growth'].tolist()
            },
            "segment_distribution": segment_data,
            "seasonality": seasonality,
            "projections": projections,
            "anomalies": anomalies[:10]  # Cap at top 10 relevant anomalies
        }
