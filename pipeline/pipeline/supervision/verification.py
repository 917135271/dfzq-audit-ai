"""Verify extracted claims against full supplied evidence context."""
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=200)
    field: str = Field(min_length=1, max_length=200)
    value: str = Field(min_length=1, max_length=4000)
    evidence: str = Field(min_length=1, max_length=32000)


class VerificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[Claim] = Field(min_length=1, max_length=8)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    supported: StrictBool
    reason: str = Field(min_length=1, max_length=1000)


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdicts: list[Verdict]


def verify_claims(body: VerificationRequest, client):
    expected = {claim.id for claim in body.claims}
    if len(expected) != len(body.claims):
        raise ValueError("Duplicate claim ID")
    prompt = Path(__file__).with_name("verification-prompt.txt").read_text("utf-8")
    result = VerificationResult.model_validate(client.chat_json(prompt, body.model_dump_json()))
    if len(result.verdicts) != len(expected) or {v.id for v in result.verdicts} != expected:
        raise ValueError("Incomplete or invalid verification")
    return result.model_dump()
