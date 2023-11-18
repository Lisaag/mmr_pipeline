import pandas as pd
import settings

def get_mean_and_sdev(df: pd.DataFrame, column: str) -> tuple[float, float]:
    mean = df[column].mean()
    sdev = df[column].std()

    return mean, sdev

def filter_outliers_95th_percentile(df: pd.DataFrame, column: str, mean: float, sdev: float) -> pd.DataFrame:
    df = df[df[column] > (mean - (2 * sdev))]
    df = df[df[column] < (mean + (2 * sdev))]
    return df

def filter_averages_1sdev(df: pd.DataFrame, column: str, mean: float, sdev: float) -> pd.DataFrame:
    df = df[df[column] > (mean - (sdev))]
    df = df[df[column] < (mean + (sdev))]
    return df

def calculate_average_shape() -> None:
    csv = pd.read_csv(settings.CSV_UNNORMALIZED_OUTPUT_PATH)

    # Regulars filtering
    mean, sdev = get_mean_and_sdev(csv, 'vertex count')
    print(f'mean: {mean}, sdv: {sdev}')

    filtered = filter_outliers_95th_percentile(csv, 'vertex count', mean, sdev)
    averages = filter_averages_1sdev(csv, 'vertex count', mean, sdev)
    outliers = csv.drop(filtered.index)
    outliers.to_csv(settings.CSV_OUTLIERS_OUTPUT_PATH, index=False)
    averages.to_csv(settings.CSV_AVERAGE_SHAPE_OUTPUT_PATH, index=False)
