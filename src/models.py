from sqlalchemy import Column, Integer, String
from src.database import Base

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    age = Column(Integer)
    gender = Column(String(10))
    phone = Column(String(20))
    email = Column(String(255), index=True, nullable=False)
    password_hash = Column(String(255))

class Hospitals(Base):
    __tablename__="hospitals"

    id = Column(Integer,primary_key=True,index=True)
    name = Column(String(100))
    phone = Column(String(20))


