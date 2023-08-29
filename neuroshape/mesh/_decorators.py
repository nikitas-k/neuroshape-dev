"""
Decorator objects
"""

import datetime
import copy

class History:
    def __init__(self, *args, savestate=False):
        self.attrs = args
        self.savestate = savestate
        
    def __call__(self, cls):
        cls._history = []
        this = self
        
        oGetter = cls.__getattr__
        def getter(self, attr):
            if attr == 'history':
                return self._history
            
            return oGetter(self, attr)
        cls.__getattr__ = getter
        
        oSetter = cls.__setattr__
        def setter(self, attr, value):
            if attr in this.attrs:
                self._history.append((datetime.datetime.now(), attr, copy.deepcopy(value) if this.savestate else value))
            
            return oSetter(self, attr, value)
        cls.__setattr__ = setter
        
        return cls