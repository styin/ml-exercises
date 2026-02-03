#### Utility Functions
##### Pickling
```python
gymnasium.utils.ezpickle.EzPickle(_*args: Any_, _**kwargs: Any_)
```
- It handles the serialization of complex objects, by saving the parameters to the `__init__` method.

**Implementation**
By inheriting from `EzPickle` and calling `EzPickle.__init__(self, ...)` inside the class constructor, the object can be accurately recreated.