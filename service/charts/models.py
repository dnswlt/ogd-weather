from pydantic import BaseModel


class Station(BaseModel):
    abbr: str
    name: str
    canton: str
