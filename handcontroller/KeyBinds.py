from pynput.keyboard import Controller as KeyboardController, Key

class InputController:
    def __init__(self, bindings=None):
        self.keyboard = KeyboardController()
        #keybinds tested on Hades
        self.gesture_to_key = {
            'pointing':      'w',       # move forward
            'peace_sign':    'a',       # move left
            'three':         'd',       # move right
            'four':          's',       # move backwards
            'open_hand':     Key.space, # dash/jump
            'fist':          'z',       # attack
            'thumbs_up':     'e',       # interact
            'finger_gun':    'q',       # cast/aim
            'ok_sign':       'r',       # special attack
            'middle_finger': 'f',       # use/call
        }

        if bindings is not None:
            self.gesture_to_key = {
                name: getattr(Key, key) if len(key) > 1 else key
                for name, key in bindings.items()
            }
        self.pressed_keys = {}

    def set_active(self, active_names):
        """Apply a complete input state, including mappings that share a key."""
        old_keys = {self.gesture_to_key[name] for name, held in self.pressed_keys.items() if held}
        new_keys = {self.gesture_to_key[name] for name in active_names if name in self.gesture_to_key}
        for key in old_keys - new_keys:
            self.keyboard.release(key)
        for key in new_keys - old_keys:
            self.keyboard.press(key)
        self.pressed_keys = {name: name in active_names for name in self.gesture_to_key}

    def set_gestures(self, gesture_name: str, is_active: bool):
        if gesture_name not in self.gesture_to_key:
            return
        key = self.gesture_to_key[gesture_name]
        was_pressed = self.pressed_keys.get(gesture_name, False)
        if is_active and not was_pressed:
            self.keyboard.press(key)
            self.pressed_keys[gesture_name] = True
        elif not is_active and was_pressed:
            self.keyboard.release(key)
            self.pressed_keys[gesture_name] = False

    def release_all(self):
        for gesture_name in list(self.pressed_keys.keys()):
            if self.pressed_keys[gesture_name]:
                self.keyboard.release(self.gesture_to_key[gesture_name])
                self.pressed_keys[gesture_name] = False
