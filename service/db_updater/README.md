# db_updater

This batch service creates and updates the database with data from Meteo Swiss.

## Profiling

```bash
$ OGD_POSTGRES_URL='postgresql+psycopg://ogd_weather@localhost:5432/ogd_weather' python3 -m cProfile -o profile.stats -m service.db_updater.app --recreate-views

$ python3 -m pstats profile.stats

profile.stats% sort cumulative
profile.stats% stats 30
```
