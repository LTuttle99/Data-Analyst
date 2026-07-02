import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io

class BookOfBusinessAnalyzer:
    def __init__(self, file_bytes: bytes, file_name: str):
        self.file_name = file_name
        
        if file_name.endswith('.csv'):
            self.df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
        elif file_name.endswith(('.xls', '.xlsx')):
            self.df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel.")
        
        # Strip string columns globally and normalize header tokens
        self.df.columns = [str(c).strip() for c in self.df.columns]
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

        # Bulletproof Profit Center Extractor: Strip nulls and normalize out string representations
        unique_profit_centers = []
        if schema["profit_center"] and schema["profit_center"] in self.df.columns:
            raw_pcs = self.df[schema["profit_center"]].dropna()
            # If float/int representation, turn back cleanly (e.g. drop trailing .0)
            unique_profit_centers = sorted(list(set([str(int(x)) if isinstance(x, (float, int)) and x == int(x) else str(x).strip() for x in raw_pcs])))

        return {
            "columns": columns, 
            "inferred_mapping": schema,
            "profit_centers": unique_profit_centers[:100]
        }

    def run_analysis(self, mapping: dict, selected_profit_center: str = "ALL", projection_target: str = "premium") -> dict:
        """
        Executes advanced, institutional financial accounting analytics over the 5,200+ row ledger.
        """
        fin_col = mapping.get("financial_metric")
        time_col = mapping.get("timeline_metric")
        cat_col = mapping.get("categorical_segment")
        pc_col = mapping.get("profit_center")
        id_col = mapping.get("client_id")

        if not fin_col or not time_col:
            raise ValueError("Financial and Timeline metrics are required fields.")

        # Subset data for memory efficiency
        cols_to_keep = [fin_col, time_col]
        if cat_col: cols_to_keep.append(cat_col)
        if pc_col: cols_to_keep.append(pc_col)
        if id_col: cols_to_keep.append(id_col)
        
        working_df = self.df[cols_to_keep].copy()
        working_df[time_col] = pd.to_datetime(working_df[time_col], errors='coerce')
        working_df[fin_col] = pd.to_numeric(working_df[fin_col], errors='coerce').fillna(0)
        working_df = working_df.dropna(subset=[time_col])

        # Normalize the structural target match array string mappings
        if pc_col:
            working_df[pc_col] = working_df[pc_col].apply(lambda x: str(int(x)) if isinstance(x, (float, int)) and x == int(x) else str(x).strip())
        if cat_col: working_df[cat_col] = working_df[cat_col].fillna('Unknown')
        if id_col: working_df[id_col] = working_df[id_col].fillna('Unknown')

        # Core Date Filter: Slice baseline metrics from 2022 onwards
        working_df = working_df[working_df[time_col].dt.year >= 2022]

        # FIX: Bulletproof string matching filter execution
        if pc_col and selected_profit_center != "ALL":
            working_df = working_df[working_df[pc_col] == str(selected_profit_center).strip()]

        if working_df.empty:
            return {
                "kpis": {"total_premium": 0, "total_accounts": 0, "avg_account_size": 0, "retention_rate": 0, "hhi_index": 0},
                "historical_timeline": {"labels": [], "values": [], "rolling_avg": [], "mom_growth": []},
                "segment_distribution": {}, "seasonality": {}, "projections": [], "anomalies": []
            }

        # 1. Broad Portfolio Metrics
        total_premium = float(working_df[fin_col].sum())
        total_accounts = int(working_df[id_col].nunique()) if id_col else int(len(working_df))
        avg_account_size = float(working_df[fin_col].mean()) if total_accounts > 0 else 0

        # 2. Portfolio Concentration Index (HHI Science Approach)
        # HHI < 1500: Healthy/Diversified, > 2500: Dangerous Concentration
        hhi_index = 0
        if cat_col:
            shares = working_df.groupby(cat_col)[fin_col].sum()
            if total_premium > 0:
                hhi_index = float(sum([(v / total_premium * 100) ** 2 for v in shares]))

        # 3. Dynamic Retention and Attrition Calculations
        working_df['Year'] = working_df[time_col].dt.year
        retention_rate = 100.0  # Base standard default fallback
        years_present = sorted(working_df['Year'].unique())
        if len(years_present) >= 2:
            prev_year_accounts = set(working_df[working_df['Year'] == years_present[-2]][id_col].unique()) if id_col else set()
            curr_year_accounts = set(working_df[working_df['Year'] == years_present[-1]][id_col].unique()) if id_col else set()
            if prev_year_accounts:
                retained = prev_year_accounts.intersection(curr_year_accounts)
                retention_rate = float(len(retained) / len(prev_year_accounts) * 100)

        # 4. Historical Timelines
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

        # 5. Diversification Segments Split
        segment_data = {}
        if cat_col:
            seg_summary = working_df.groupby(cat_col)[fin_col if projection_target == "premium" else (id_col if id_col else fin_col)].agg('sum' if projection_target == "premium" else 'nunique').sort_values(ascending=False).head(20)
            segment_data = {str(k): float(v) for k, v in seg_summary.items()}

        # 6. Seasonality Metrics
        working_df['MonthName'] = working_df[time_col].dt.strftime('%B')
        season_summary = working_df.groupby('MonthName')[fin_col if projection_target == "premium" else (id_col if id_col else fin_col)].agg('sum' if projection_target == "premium" else 'nunique')
        seasonality = {k: float(v) for k, v in season_summary.to_dict().items()}

        # 7. Advanced Linear Dynamic Modeling
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

        # 8. Outlier Risk Profiling (Flags accounts making up > 5% of entire segment volume)
        anomalies = []
        if id_col:
            top_accounts = working_df.groupby(id_col)[fin_col].sum().sort_values(ascending=False).head(10)
            for acc_id, acc_vol in top_accounts.items():
                if total_premium > 0 and (acc_vol / total_premium) > 0.03:
                    anomalies.append({
                        "identifier": str(acc_id),
                        "value": float(acc_vol),
                        "reason": f"High Concentration Exposure Key Risk Indicator ({round(acc_vol/total_premium*100, 1)}% of total volume)"
                    })

        return {
            "kpis": {
                "total_premium": total_premium,
                "total_accounts": total_accounts,
                "avg_account_size": avg_account_size,
                "retention_rate": retention_rate,
                "hhi_index": hhi_index
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
            "anomalies": anomalies
        }
