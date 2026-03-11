from pynput.keyboard import Controller as KeyboardController, Key

class InputController:
    def __init__(self):
        self.keyboard = KeyboardController()

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

        self.pressed_keys = {}

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