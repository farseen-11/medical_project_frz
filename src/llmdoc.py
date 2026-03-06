import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate


# Load env
load_dotenv()

# -------------------------
# Pydantic schema
# -------------------------
class DoctorRecommendation(BaseModel):
    doctor: str = Field(description="Recommended specialist doctor")
    urgency: str = Field(description="Urgency level: Low, Moderate, High")
    reason: str = Field(description="Short medical reasoning")


# -------------------------
# Gemini model
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2
)


parser = PydanticOutputParser(pydantic_object=DoctorRecommendation)



prompt = PromptTemplate(
    template="""
    You are an expert medical triage assistant.

    Based on the following Diseases, recommend:
    1. The most relevant and severe one specialist doctor available in kerala like "General medicine/optometry/cardiologist/neurologist/gaestrologist/orthopedics/dermatologist like that,give any one.
    2. Don't return multiple doctors.
    3. Urgency level.
    4. Short reasoning.

    Diseases: {diseases}

    {format_instructions}
    """,
    input_variables=["diseases"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


def get_doctor_from_llm(diseases: list):
    chain = prompt | llm | parser
    return chain.invoke({"diseases": ", ".join(diseases)})
