from datetime import timedelta

VOL_SHORT_COL = "returns_30d_vol"
VOL_LONG_COL = "returns_365d_vol"
VOL_FORECAST_COL = "vol_forecast"
VOL_TARGET_COL = "volatility_target"

SCALED_SIGNAL_COL = "signal_scaled"
ABS_SIGNAL_AVG_COL = "abs_{signal}_cross_section_180d_ema"
RANK_COL = "{signal}_rank"
IDM_COL = "idm"
IDM_30D_EMA_COL = "idm_30d_ema"
FDM_COL = "fdm"
FDM_30D_EMA_COL = "fdm_30d_ema"
DM_COL = "dm_combined"
DM_30D_EMA_COL = "dm_30d_ema"
POSITION_SCALING_FACTOR_COL = "position_scaling_factor"

NUM_UNIQUE_ASSETS_COL = "num_unique_assets"
NUM_LONG_ASSETS_COL = "num_long_assets"
NUM_SHORT_ASSETS_COL = "num_short_assets"
NUM_KEPT_ASSETS_COL = "num_kept_assets"

MAX_ABS_POSITION_SIZE_COL = "max_abs_position_size"
NUM_OPEN_LONG_POSITIONS_COL = "num_open_long_positions"
NUM_OPEN_SHORT_POSITIONS_COL = "num_open_short_positions"
NUM_OPEN_POSITIONS_COL = "num_open_positions"

IDM_REFRESH_PERIOD = timedelta(days=90)
