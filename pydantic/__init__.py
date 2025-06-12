import json

class ValidationError(Exception):
    pass

class BaseModel:
    def __init__(self, **data):
        anns = getattr(self.__class__, '__annotations__', {})
        for name in anns:
            if name not in data:
                raise ValidationError(f"Missing field: {name}")
        for k, v in data.items():
            setattr(self, k, v)

    def __eq__(self, other):
        return self.__dict__ == getattr(other, '__dict__', {})

    def json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def parse_raw(cls, data):
        return cls(**json.loads(data))

    @classmethod
    def model_validate_json(cls, data):
        return cls.parse_raw(data)

    @classmethod
    def schema_json(cls):
        return "{}"

class ConfigDict(dict):
    pass
