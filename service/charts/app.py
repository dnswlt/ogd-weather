import altair as alt
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from . import charts


class ChartRequest(BaseModel):
    city: str
    chart_type: str


# Always create the app, we're running this thing with uvicorn ONLY.
app = FastAPI()


@app.exception_handler(charts.StationNotFoundError)
async def station_not_found_handler(request, exc: charts.StationNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )


@app.exception_handler(charts.NoDataError)
async def station_not_found_handler(request, exc: charts.NoDataError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )


@app.get("/stations/{station_abbr}/charts/{chart_type}")
async def get_chart(station_abbr: str, chart_type: str, month: int = 6):
    if chart_type == "temperature":
        return charts.temperature_chart(station_abbr, month=month)
    elif chart_type == "precipitation":
        return charts.precipitation_chart(station_abbr, month=month)

    valid_charts = ["temperature", "precipitation"]
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid chart type (must be one of [{','.join(valid_charts)}])",
    )


@app.get("/stations")
async def list_stations(cantons: str = None):
    if cantons:
        cantons = cantons.split(",")
    stations = charts.list_stations(cantons=cantons)
    return {
        "stations": stations,
    }
