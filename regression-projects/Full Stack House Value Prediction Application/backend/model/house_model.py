from pydantic import BaseModel

class House(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: int = 3
    total_bedrooms: int = 3
    population: int 
    households: int = 10
    median_income: float
    ocean_proximity: object
    

