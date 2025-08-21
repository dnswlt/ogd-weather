# Filename for the sqlite3 database.
DATABASE_FILENAME = "swissmetnet.sqlite"

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

# Precipitation
PRECIP_HOURLY_MM = "rre150h0"
PRECIP_DAILY_MM = "rre150d0"
PRECIP_MONTHLY_MM = "rre150m0"

# Wind
WIND_SPEED_HOURLY_MEAN = "fkl010h0"
WIND_SPEED_DAILY_MEAN = "fkl010d0"
WIND_SPEED_MONTHLY_MEAN = "fkl010m0"

WIND_DIRECTION_HOURLY_MEAN = "dkl010h0"
WIND_DIRECTION_DAILY_MEAN = "dkl010d0"

GUST_PEAK_HOURLY_MAX = "fkl010h1"
GUST_PEAK_DAILY_MAX = "fkl010d1"
GUST_PEAK_MONTHLY_MAX = "fkl010m1"

# Atmospheric pressure
ATM_PRESSURE_DAILY_MEAN = "prestad0"
ATM_PRESSURE_HOURLY_MEAN = "prestah0"
ATM_PRESSURE_MONTHLY_MEAN = "prestam0"

# Humidity
REL_HUMIDITY_HOURLY_MEAN = "ure200h0"
REL_HUMIDITY_DAILY_MEAN = "ure200d0"
REL_HUMIDITY_MONTHLY_MEAN = "ure200m0"

# Sunshine
SUNSHINE_HOURLY_MINUTES = "sre000h0"
SUNSHINE_DAILY_MINUTES = "sre000d0"
SUNSHINE_MONTHLY_MINUTES = "sre000m0"

SUNSHINE_DAILY_PCT_OF_MAX = "sremaxdv"

# Snow
# According to https://opendatadocs.meteoswiss.ch/general/faq
# the official snow depth measurement is the manual one (as of Aug 2025).
SNOW_DEPTH_DAILY_CM = "htoautd0"
SNOW_DEPTH_MANUAL_DAILY_CM = "hto000d0"
FRESH_SNOW_MANUAL_DAILY_CM = "hns000d0"

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

# Derived metric names (prefix DX_):
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
