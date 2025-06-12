import json

class ValidationError(Exception):
    pass

class BaseModel:
    def __init__(self, **data):
        """Populate fields declared in ``__annotations__`` allowing defaults."""
        anns = getattr(self.__class__, '__annotations__', {})
        # First, set annotated fields either from provided data or defaults
        for name in anns:
            if name in data:
                value = data.pop(name)
                setattr(self, name, value)
            elif hasattr(self.__class__, name):
                # honour class level defaults (e.g. ``field: int = 1``)
                setattr(self, name, getattr(self.__class__, name))
            else:
                raise ValidationError(f"Missing field: {name}")

        # Store any extra fields directly on the instance
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
