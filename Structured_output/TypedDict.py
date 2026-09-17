from typing import TypedDict # TypedDict define the type the dict must elements must be of, it just warns about the type do not enforce any validation even if the type of the elements of the dict is not what the TypedDict mention the code will still work 

class Person(TypedDict):
    name:str
    age:int


new_person:Person={"name":"Ayush","age":23}

print(new_person)