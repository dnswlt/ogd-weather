"""Contains database and OGD SwissMeteo source file constants."""

# Filename for the sqlite3 database.
SQLITE_DB_FILENAME = "swissmetnet.sqlite"

# Column names for weather measurements.

# Temperature
TEMP_HOURLY_MIN = "tre200hn"
TEMP_HOURLY_MEAN = "tre200h0"
TEMP_HOURLY_MAX = "tre200hx"
TEMP_DAILY_MIN = "tre200dn"
TEMP_DAILY_MEAN = "tre200d0"
TEMP_DAILY_MAX = "tre200dx"
TEMP_MONTHLY_MIN = "tre200mn"
TEMP_MONTHLY_MEAN = "tre200m0"
TEMP_MONTHLY_MAX = "tre200mx"
TEMP_HOM_DAILY_MIN = "ths200dn"
TEMP_HOM_DAILY_MEAN = "ths200d0"
TEMP_HOM_DAILY_MAX = "ths200dx"
TEMP_HOM_DAILY_MIN_DEV_FROM_NORM_9120 = "th91dndv"
TEMP_HOM_DAILY_MEAN_DEV_FROM_NORM_9120 = "th9120dv"
TEMP_HOM_DAILY_MAX_DEV_FROM_NORM_9120 = "th91dxdv"
TEMP_HOM_MONTHLY_MEAN = "ths200m0"
TEMP_HOM_MONTHLY_MEAN_OF_DAILY_MIN = "ths2dymn"
TEMP_HOM_MONTHLY_MEAN_OF_DAILY_MAX = "ths2dymx"
TEMP_HOM_MONTHLY_MIN = "thl200mn"
TEMP_HOM_MONTHLY_MAX = "thl200mx"
TEMP_HOM_MONTHLY_MEAN_DEV_FROM_NORM_9120 = "th9120mv"
TEMP_HOM_MONTHLY_MEAN_OF_DAILY_MIN_DEV_FROM_NORM_9120 = "th91dnmv"
TEMP_HOM_MONTHLY_MEAN_OF_DAILY_MAX_DEV_FROM_NORM_9120 = "th91dxmv"
TEMP_HOM_ANNUAL_MEAN = "ths200y0"
TEMP_HOM_ANNUAL_MEAN_OF_DAILY_MIN = "ths2dyyn"
TEMP_HOM_ANNUAL_MEAN_OF_DAILY_MAX = "ths2dyyx"
TEMP_HOM_ANNUAL_MIN = "thl200yn"
TEMP_HOM_ANNUAL_MAX = "thl200yx"
TEMP_HOM_ANNUAL_MEAN_DEV_FROM_NORM_9120 = "th9120yv"
TEMP_HOM_ANNUAL_MEAN_OF_DAILY_MIN_DEV_FROM_NORM_9120 = "th91dnyv"
TEMP_HOM_ANNUAL_MEAN_OF_DAILY_MAX_DEV_FROM_NORM_9120 = "th91dxyv"
# Special Temperature Days (Monthly Count)
FROST_DAYS_HOM_MONTHLY_COUNT = "ths00nm0"  # Min temp < 0°C
ICE_DAYS_HOM_MONTHLY_COUNT = "ths00xm0"  # Max temp < 0°C
TROPICAL_NIGHTS_HOM_MONTHLY_COUNT = "ths20nm0"
SUMMER_DAYS_HOM_MONTHLY_COUNT = "ths25xm0"  # Max temp >= 25°C
HEAT_DAYS_HOM_MONTHLY_COUNT = "ths30xm0"  # Max temp >= 30°C
VERY_HOT_DAYS_HOM_MONTHLY_COUNT = "ths35xm0"  # Max temp >= 35°C
# Special Temperature Days (Annual Count)
FROST_DAYS_HOM_ANNUAL_COUNT = "ths00ny0"
ICE_DAYS_HOM_ANNUAL_COUNT = "ths00xy0"
TROPICAL_NIGHTS_HOM_ANNUAL_COUNT = "ths20ny0"
SUMMER_DAYS_HOM_ANNUAL_COUNT = "ths25xy0"
HEAT_DAYS_HOM_ANNUAL_COUNT = "ths30xy0"
VERY_HOT_DAYS_HOM_ANNUAL_COUNT = "ths35xy0"


# Precipitation
PRECIP_HOURLY_MM = "rre150h0"
PRECIP_DAILY_MM = "rre150d0"
PRECIP_MONTHLY_MM = "rre150m0"
PRECIP_ANNUAL_MM = "rre150y0"
PRECIP_HOM_MONTHLY_MM = "rhs150m0"
PRECIP_DAYS_GT_1MM_HOM_MONTHLY_COUNT = "rhs001m0"
PRECIP_HOM_MONTHLY_MM_VS_NORM_9120 = "rh9120mv"
PRECIP_HOM_ANNUAL_MM = "rhs150y0"
PRECIP_DAYS_GT_1MM_HOM_ANNUAL_COUNT = "rhs001y0"
PRECIP_HOM_ANNUAL_MM_VS_NORM_9120 = "rh9120yv"
PRECIP_DAYS_HOM_ANNUAL_LONGEST_SPELL = "rhs001yx"
DRY_DAYS_HOM_ANNUAL_LONGEST_SPELL = "rhsdryyx"  # Dry day is precip < 1mm


# Wind
WIND_SPEED_HOURLY_MEAN = "fkl010h0"
WIND_SPEED_DAILY_MEAN = "fkl010d0"
WIND_SPEED_MONTHLY_MEAN = "fkl010m0"
WIND_SPEED_HOM_MONTHLY_MEAN = "fhs010m0"
WIND_SPEED_HOM_MONTHLY_MEAN_VS_NORM_9120 = "fh9120mv"
WIND_SPEED_HOM_ANNUAL_MEAN = "fhs010y0"
WIND_SPEED_HOM_ANNUAL_MEAN_VS_NORM_9120 = "fh9120yv"

WIND_DIRECTION_HOURLY_MEAN = "dkl010h0"
WIND_DIRECTION_DAILY_MEAN = "dkl010d0"

GUST_PEAK_HOURLY_MAX = "fkl010h1"
GUST_PEAK_DAILY_MAX = "fkl010d1"
GUST_PEAK_MONTHLY_MAX = "fkl010m1"


# Atmospheric pressure
ATM_PRESSURE_DAILY_MEAN = "prestad0"
ATM_PRESSURE_HOURLY_MEAN = "prestah0"
ATM_PRESSURE_MONTHLY_MEAN = "prestam0"
ATM_PRESSURE_HOM_MONTHLY_MEAN = "phsstam0"
ATM_PRESSURE_QFE_HOM_MONTHLY_MEAN_DEV_FROM_NORM_9120 = "ph9120mv"
ATM_PRESSURE_HOM_ANNUAL_MEAN = "phsstay0"
ATM_PRESSURE_QFE_HOM_ANNUAL_MEAN_DEV_FROM_NORM_9120 = "ph9120yv"

# Humidity
REL_HUMIDITY_HOURLY_MEAN = "ure200h0"
REL_HUMIDITY_DAILY_MEAN = "ure200d0"
REL_HUMIDITY_MONTHLY_MEAN = "ure200m0"

# Sunshine
SUNSHINE_HOURLY_MINUTES = "sre000h0"
SUNSHINE_DAILY_MINUTES = "sre000d0"
SUNSHINE_MONTHLY_MINUTES = "sre000m0"
SUNSHINE_DAILY_PCT_OF_MAX = "sremaxdv"
SUNSHINE_HOM_MONTHLY_MINUTES = "shs000m0"
SUNSHINE_HOM_MONTHLY_MINUTES_VS_NORM_9120 = "sh9120mv"
SUNSHINE_HOM_ANNUAL_MINUTES = "shs000y0"
SUNSHINE_HOM_ANNUAL_MINUTES_VS_NORM_9120 = "sh9120yv"

# Vapor Pressure
VAPOR_PRESSURE_HOURLY_MEAN = "pva200h0"
VAPOR_PRESSURE_HOM_MONTHLY_MEAN = "pva2hsm0"
VAPOR_PRESSURE_HOM_MONTHLY_MEAN_DEV_FROM_NORM_9120 = "pv9120mv"
VAPOR_PRESSURE_HOM_ANNUAL_MEAN = "pva2hsy0"
VAPOR_PRESSURE_HOM_ANNUAL_MEAN_DEV_FROM_NORM_9120 = "pv9120yv"

# Global Radiation
GLOBAL_RADIATION_HOURLY_MEAN = "gre000h0"
GLOBAL_RADIATION_HOM_MONTHLY_MEAN = "ghs000m0"
GLOBAL_RADIATION_HOM_MONTHLY_MEAN_VS_NORM_9120 = "gh9120mv"
GLOBAL_RADIATION_HOM_ANNUAL_MEAN = "ghs000y0"
GLOBAL_RADIATION_HOM_ANNUAL_MEAN_VS_NORM_9120 = "gh9120yv"

# Degree Days (Heating/Cooling)
HEATING_DEGREE_DAYS_SIA_HOM_MONTHLY_SUM = "xhs000m0"  # Base 12/20
HEATING_DEGREE_DAYS_ATD12_HOM_MONTHLY_SUM = "xhs012m0"
COOLING_DEGREE_DAYS_HOM_MONTHLY_SUM = "xhs00om0"
HEATING_DEGREE_DAYS_SIA_HOM_ANNUAL_SUM = "xhs000y0"
HEATING_DEGREE_DAYS_ATD12_HOM_ANNUAL_SUM = "xhs012y0"
COOLING_DEGREE_DAYS_HOM_ANNUAL_SUM = "xhs00oy0"


# Snow
# According to https://opendatadocs.meteoswiss.ch/general/faq
# the official snow depth measurement is the manual one (as of Aug 2025).
SNOW_DEPTH_DAILY_CM = "htoautd0"
SNOW_DEPTH_MAN_DAILY_CM = "hto000d0"
SNOW_DEPTH_MAN_MONTHLY_CM = "hto000m0"
SNOW_DEPTH_MAN_ANNUAL_CM = "hto000y0"
FRESH_SNOW_MAN_DAILY_CM = "hns000d0"
FRESH_SNOW_MAN_MONTHLY_CM = "hns000m0"
FRESH_SNOW_MAN_ANNUAL_CM = "hns000y0"

# Map SwissMetNet parameter names to readable names to use at the API level
# (e.g. when returning summary stats for variables).
VARIABLE_API_NAMES = {
    TEMP_DAILY_MIN: "temperature_daily_min",
    TEMP_DAILY_MAX: "temperature_daily_max",
    TEMP_DAILY_MEAN: "temperature_daily_mean",
    PRECIP_DAILY_MM: "precipitation_daily_millimeters",
    WIND_SPEED_DAILY_MEAN: "wind_speed_daily_mean",
    WIND_DIRECTION_DAILY_MEAN: "wind_direction_daily_mean",
    SUNSHINE_DAILY_MINUTES: "sunshine_daily_minutes",
    GUST_PEAK_DAILY_MAX: "gust_peak_daily_max",
    ATM_PRESSURE_DAILY_MEAN: "atm_pressure_daily_mean",
    REL_HUMIDITY_DAILY_MEAN: "rel_humidity_daily_mean",
    SUNSHINE_DAILY_MINUTES: "sunshine_daily_minutes",
    SUNSHINE_DAILY_PCT_OF_MAX: "sunshine_daily_pct_of_max",
    SNOW_DEPTH_DAILY_CM: "snow_depth_daily_cm",
}

############################################################
# Derived metric names (prefix DX_):
############################################################

# Day counts
DX_SUNNY_DAYS_ANNUAL_COUNT = "sunny_days_annual_count"
DX_SUMMER_DAYS_ANNUAL_COUNT = "summer_days_annual_count"
DX_RAIN_DAYS_ANNUAL_COUNT = "rain_days_annual_count"
DX_FROST_DAYS_ANNUAL_COUNT = "frost_days_annual_count"
DX_TROPICAL_NIGHTS_ANNUAL_COUNT = "tropical_nights_annual_count"
DX_GROWING_DEGREE_DAYS_ANNUAL_SUM = "growing_degree_days_annual_sum"

DX_SUNNY_DAYS_MONTHLY_COUNT = "sunny_days_monthly_count"
DX_SUMMER_DAYS_MONTHLY_COUNT = "summer_days_monthly_count"
DX_FROST_DAYS_MONTHLY_COUNT = "frost_days_monthly_count"
DX_TROPICAL_NIGHTS_MONTHLY_COUNT = "tropical_nights_monthly_count"
DX_GROWING_DEGREE_DAYS_MONTHLY_SUM = "growing_degree_days_monthly_sum"

# Wind
DX_WIND_SPEED_DAILY_MAX_OF_HOURLY_MEAN = "wind_speed_dxh0"
DX_WIND_SPEED_DAYTIME_DAILY_MAX_OF_HOURLY_MEAN = "wind_speed_dt_dxh0"
DX_GUST_PEAK_DAYTIME_DAILY_MAX = "gust_peak_dt_dx"
DX_GUST_FACTOR_DAILY_MEAN = "gust_factor_d0"
DX_GUST_FACTOR_MONTHLY_MEAN = "gust_factor_m0"
DX_MODERATE_BREEZE_DAYS_MONTHLY_COUNT = "moderate_breeze_days_m0"
DX_STRONG_BREEZE_DAYS_MONTHLY_COUNT = "strong_breeze_days_m0"

# Generic column: Value counts (for aggregate data: indicates
# the number of values or measurements from which the aggregates
# were calculated).
DX_VALUE_COUNT = "value_count"

# For the wind rose histogram. For simplicity, these variables
# do not contain an interval qualifier.
DX_WIND_DIR_N_COUNT = "wind_dir_n_count"
DX_WIND_DIR_NE_COUNT = "wind_dir_ne_count"
DX_WIND_DIR_E_COUNT = "wind_dir_e_count"
DX_WIND_DIR_SE_COUNT = "wind_dir_se_count"
DX_WIND_DIR_S_COUNT = "wind_dir_s_count"
DX_WIND_DIR_SW_COUNT = "wind_dir_sw_count"
DX_WIND_DIR_W_COUNT = "wind_dir_w_count"
DX_WIND_DIR_NW_COUNT = "wind_dir_nw_count"
DX_WIND_DIR_TOTAL_COUNT = "wind_dir_total_count"
DX_WIND_DIR_COUNT_MAP = {
    "N": DX_WIND_DIR_N_COUNT,
    "NE": DX_WIND_DIR_NE_COUNT,
    "E": DX_WIND_DIR_E_COUNT,
    "SE": DX_WIND_DIR_SE_COUNT,
    "S": DX_WIND_DIR_S_COUNT,
    "SW": DX_WIND_DIR_SW_COUNT,
    "W": DX_WIND_DIR_W_COUNT,
    "NW": DX_WIND_DIR_NW_COUNT,
}

# Precipitation
DX_PRECIP_DAYTIME_DAILY_MM = "precip_dt_d0"

# Air humidity
DX_VAPOR_PRESSURE_DAYTIME_DAILY_MAX_OF_HOURLY_MEAN = "vapor_pressure_dt_dxh0"

# Total precipitation per given time slice and year.
# E.g., in monthly summary stats for the 1991-2020 reference
# period, this derived metric holds the monthly precipitation.
DX_PRECIP_TOTAL = "dx_precip_total"

DX_SOURCE_DATE_RANGE = "source_date_range"


# Aggregation names in STATION_VAR_SUMMARY_STATS_TABLE_NAME
AGG_NAME_REF_1991_2020 = "ref_1991_2020"

# Time slice value for aggregations that have no actual time slice.
TS_ALL = "*"


def ts_month(month: int) -> str:
    """Returns the time_slice string for the given month."""
    return f"{month:02d}"
