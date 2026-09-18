from pydantic import BaseModel,EmailStr,Field
from typing import Optional


class Student(BaseModel):
    name: str = "XYZ" # default value
    age:Optional[int] = None # optional value will None if not set
    email:EmailStr  # with pydantic we can verify emails (built in validation)
    cgpa:float =Field(gt=0,lt=10,default=4,description='A decimal value representing the cgpa of the student')
    
new_student = {"age":23,"cgpa":8,"email":"ayushsharma7103@gmail.com"}

student = Student(**new_student)

print(student)

# convert this output of json 

student_json = student.model_dump_json()

print("Student Json"," ",student_json)