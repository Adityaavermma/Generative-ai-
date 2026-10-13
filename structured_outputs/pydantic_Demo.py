from pydantic import BaseModel,EmailStr,Field
from typing import Optional
class Student(BaseModel):
    name:str='aditya'
    age:Optional[int]=None
    email: EmailStr
    cgpa:float= Field(gt=0,lt=10)
new_student=Student(name='aditya', age=20, email='aditya@example.com', cgpa=8.5)
student=(new_student)
print(student,dict(student))
student_json=student.model_dump_json()
print(student_json)