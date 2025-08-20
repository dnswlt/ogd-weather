import io
import pandas as pd
import requests


def print_params(params_file, datafiles, lang="de"):
    r = requests.get(params_file, timeout=5)
    r.raise_for_status()
    df_p = pd.read_csv(io.BytesIO(r.content), encoding="cp1252", sep=";")
    for f in datafiles:
        print(f"Parameters in {f}:\n")
        r = requests.get(f, timeout=5)
        df = pd.read_csv(io.BytesIO(r.content), encoding="cp1252", sep=";")
        params = df_p[df_p["parameter_shortname"].isin(df.columns)]
        for n, d in params[
            ["parameter_shortname", f"parameter_description_{lang}"]
        ].itertuples(index=False):
            print(n, d)
        print("\n")


def print_smn_params():
    files = [
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ber/ogd-smn_ber_h_recent.csv",
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ber/ogd-smn_ber_d_recent.csv",
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ber/ogd-smn_ber_m.csv",
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ber/ogd-smn_ber_y.csv",
    ]
    params = (
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ogd-smn_meta_parameters.csv"
    )
    print_params(params, files)


def print_nbcn_params():
    files = [
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nbcn/ber/ogd-nbcn_ber_d_recent.csv",
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nbcn/ber/ogd-nbcn_ber_m.csv",
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nbcn/ber/ogd-nbcn_ber_y.csv",
    ]
    params = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nbcn/ogd-nbcn_meta_parameters.csv"
    print_params(params, files)


def print_nime_params():
    files = [
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nime/ber/ogd-nime_ber_d_recent.csv",
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nime/ber/ogd-nime_ber_m.csv",
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nime/ber/ogd-nime_ber_y.csv",
    ]
    params = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nime/ogd-nime_meta_parameters.csv"
    print_params(params, files)


def print_stations(dataset: str):
    r = requests.get(
        f"https://data.geo.admin.ch/ch.meteoschweiz.ogd-{dataset}/ogd-{dataset}_meta_stations.csv",
        timeout=5,
    )
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), encoding="cp1252", sep=";")

    print(f"Weather stations for dataset '{dataset}':\n")

    for abbr, name, canton in df[
        ["station_abbr", "station_name", "station_canton"]
    ].itertuples(index=False):
        print(abbr, name, canton)

    print("\n")
    return set(df["station_abbr"])


def main():
    print_smn_params()
    print_nbcn_params()
    print_nime_params()
    print_stations("smn")
    print_stations("nbcn")


if __name__ == "__main__":
    main()
