
_CLASS_NAME = ""

def get_class_name():
    global _CLASS_NAME
    return _CLASS_NAME

def set_class_name(name):
    global _CLASS_NAME
    _CLASS_NAME = name
    print(f"Set class name to {get_class_name()}")