ALL_CLASS_NAMES = ["can"  , "fabric"  , "fruit_jelly"  , "rice"  , "sheet_metal"  , "vial"  , "wallplugs"  , "walnuts"]
_CLASS_NAME = ""
_IMG_CUT_NR = 1

def get_img_cut_nr():
    return _IMG_CUT_NR

def set_img_cut_nr(cut_nr):
    global _IMG_CUT_NR
    _IMG_CUT_NR = cut_nr
    print(f"Set img cut nr to {get_img_cut_nr()}")

def get_class_name():
    global _CLASS_NAME
    return _CLASS_NAME

def auto_set_img_cut_nr(name):
    assert name in ALL_CLASS_NAMES, f"ERROR class name {name}"
    if name in ["can","rice", "wallplugs"]:
        set_img_cut_nr(2)

def set_class_name(name):
    global _CLASS_NAME
    _CLASS_NAME = name
    auto_set_img_cut_nr(name)
    assert name in ALL_CLASS_NAMES, f"ERROR class name {name}"
    print(f"Set class name to {get_class_name()}")
