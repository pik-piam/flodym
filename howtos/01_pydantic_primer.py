# %% [markdown]
# # A `pydantic` Primer
#
# Most classes in sodym are based on the `pydantic` library, which provides safety in data handling.
# More specifically, most classes in sodym are `pydanctic.BaseModel`s.
#
# Their main property is that the arguments passed to the class initialization are replaced by so-called fields.
# While a normal class would be defined and initialized like this:


# %%
class Person:
    def __init__(self, name):
        self.name = name


person_a = Person("Adam")
print(person_a.name)

# %% [markdown]
# this would look slightly different for pydantic, where the `__init__` function is pre-implemented:

# %%
from pydantic import BaseModel


class Person(BaseModel):
    name: str


person_a = Person(name="Adam")
print(person_a.name)


# %% [markdown]
# There are two major differences to normal classes:
#
# - The input is type-checked. In the example, `name` has to be a string, and e.g. initializing Person like this: `Person(name=2)` will lead to an error.
# - Normal classes do not guarantee that parameters passed to init end up as class attributes. So the following will lead to an error:
#
#


# %%
class Person:
    def __init__(self, first_name: str, last_name: str):
        self.initials = f"{first_name[0]}.{last_name[0]}."


person_a = Person(first_name="Adam", last_name="Smith")

try:
    print(person_a.first_name)
except AttributeError as e:
    print(f"AttributeError: {e}")


# %% [markdown]
# In pydantic, what you put in at initialization is always what you get as attributes.
#
# This creates transparency.
#
# But on the other hand, it can be more convenient to have some operations done on initialization.
#
# Therefore, sodym sometimes offers class methods to create objects from different arguments, like for the example above:


# %%
class Person(BaseModel):

    initials: str

    @classmethod
    def from_name(cls, first_name: str, last_name: str):
        initials = f"{first_name[0]}.{last_name[0]}."
        return cls(initials=initials)


person_a = Person.from_name(first_name="Adam", last_name="Smith")
print(person_a.initials)

# %% [markdown]
# For further information on pydantic, please refer to the pydantic documentation.
