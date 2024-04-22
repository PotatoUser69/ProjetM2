import pandas as pd

def is_time_series(csv_file_path):
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)
        
        # Check if there's a column representing time
        for column in df.columns:
            try:
                # Attempt to convert the values in the column to datetime
                pd.to_datetime(df[column])
                return True  # If successful, it's likely a time series
            except ValueError:
                pass
        return False  # If none of the columns can be converted to datetime, it's not a time series
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False

csv_file_path = "C:\My Projects\ProjetM2-1\datatime_date.csv"
is_time_series_result = is_time_series(csv_file_path)
print(f"Is the dataset a time series? {is_time_series_result}")
