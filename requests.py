class Response:
    def __init__(self, text=""):
        self.text = text
    def raise_for_status(self):
        pass

def get(*args, **kwargs):
    raise RuntimeError("Network disabled")
