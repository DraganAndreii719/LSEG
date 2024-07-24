import os
import random
import pandas as pd
from datetime import datetime, timedelta

def read_csv_files(data_pth):  # Read file
    stk = {}
    try:
        for exch in os.listdir(data_pth):  
            exch_path = os.path.join(data_pth, exch)  
            if os.path.isdir(exch_path):  
                for fl in os.listdir(exch_path):  
                    if fl.endswith('.csv'):  
                        stk_id = fl.split('.')[0]  
                        fl_path = os.path.join(exch_path, fl)  
                        try:
                            df = pd.read_csv(fl_path, names=['Stock-ID', 'Timestamp', 'Stock Price'], header=None)
                            if df.empty:
                                print(f"Warning: {fl_path} is empty and will be skipped.")
                                continue
                            if stk_id not in stk:
                                stk[stk_id] = []
                            stk[stk_id].append(df)
                        except pd.errors.EmptyDataError:
                            print(f"Error: {fl_path} is empty or not a valid CSV.")
                       
    except FileNotFoundError:
        print(f"Error: The directory {data_pth} does not exist.")
        print(f"Error: Permission denied for directory {data_pth}.")
    
    return stk

def ten_consec(stk): #Extract ten consec points
    data = []
    for stk_id, dfs in stk.items():
        for df in dfs:
            try:
                if len(df) < 10:
                    continue
                start_idx = random.randint(0, len(df) - 10)
                data.append((stk_id, df.iloc[start_idx:start_idx + 10]))
            except Exception as e:
                print(f"Unexpected error while processing {stk_id}: {e}")
    return data

def predict_next_vals(data): #Predict Values
    preds = []
    for stk_id, df in data:
    
            vals = df['Stock Price'].values
            if len(vals) < 10:
                continue
            n = vals[-1]
            n1 = sorted(vals)[-2]
            n2 = n + 0.5 * (n1 - n)
            n3 = n2 + 0.25 * (n1 - n2)
            
            last_ts = df['Timestamp'].iloc[-1]
            new_tss = gen_fut_timestamps(last_ts, 3)
            
            preds.append({
                'Stock-ID': stk_id,
                'Original Data': df,
                'Timestamp': new_tss,
                'Stock Price': [n1, n2, n3]
            })
    return preds

def gen_fut_timestamps(last_ts, num_days):
    dt_format = "%d-%m-%Y"  
    last_dt = datetime.strptime(last_ts, dt_format)
    fut_tss = [(last_dt + timedelta(days=i)).strftime(dt_format) for i in range(1, num_days + 1)]
    return fut_tss


def output_preds(preds, output_pth):
    try:
        os.makedirs(output_pth, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory {output_pth}: {e}")
        return

    for pred in preds:
        try:
            orig_data = pred['Original Data'].copy()  # Make a copy to avoid SettingWithCopyWarning
            orig_data.loc[:, 'Prediction'] = 'Original'  # Use .loc to modify the DataFrame
            
            pred_df = pd.DataFrame({
                'Stock-ID': [pred['Stock-ID']] * 3,
                'Timestamp': pred['Timestamp'],
                'Stock Price': pred['Stock Price'],
                'Prediction': ['Predicted'] * 3
            })
            
            comb_df = pd.concat([orig_data, pred_df], ignore_index=True)
            output_file = os.path.join(output_pth, f"{pred['Stock-ID']}_predictions.csv")
            comb_df.to_csv(output_file, index=False)
        except PermissionError:
            print(f"Permission denied while writing to {output_file}")
        except Exception as e:
            print(f"Unexpected error while writing predictions for stock {pred['Stock-ID']}: {e}")

def main(data_pth, output_pth):
    
    stk = read_csv_files(data_pth)
    if not stk:
            print(f"No data found in {data_pth}. Exiting.")
            return
    data = ten_consec(stk)
    if not data:
            print("No sufficient data points to generate predictions. Exiting.")
            return
    preds = predict_next_vals(data)
    if not preds:
            print("No predictions generated. Exiting.")
            return
    output_preds(preds, output_pth)


if __name__ == "__main__":
    data_pth = "stock"
    output_pth = "output"
    main(data_pth, output_pth)
