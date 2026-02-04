from pydantic import BaseModel
from typing import Union

# Form-filling for profile and event(s)
class Attribute(BaseModel):
    name: str
    value: Union[str, list[str]]

class Event(BaseModel):
    event: str
    attributes: list[Attribute]

class ProfileAndEvents(BaseModel):
    profile: list[Attribute]
    events: list[Event]

# Attribute grouping
class AttributeCluster(BaseModel):
    cluster_name: str
    attributes: list[int]

class GroupedAttributes(BaseModel):
    clusters: list[AttributeCluster]