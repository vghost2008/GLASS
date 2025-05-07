ALL_CLASS_NAMES = ["can"  , "fabric"  , "fruit_jelly"  , "rice"  , "sheet_metal"  , "vial"  , "wallplugs"  , "walnuts"]
_CLASS_NAME = ""

def get_class_name():
    global _CLASS_NAME
    return _CLASS_NAME

def set_class_name(name):
    global _CLASS_NAME
    _CLASS_NAME = name
    assert name in ALL_CLASS_NAMES, f"ERROR class name {name}"
    print(f"Set class name to {get_class_name()}")