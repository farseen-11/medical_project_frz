from pydantic import BaseModel,EmailStr

class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str
    phone: str
    email: EmailStr
    password: str

class Signing(BaseModel):
    name:str
    phone:str
    password:str

class SymptomInput(BaseModel):
    symptoms: str
    top_k: int = 5

class Number(BaseModel):
    name:str
    p_number:str
