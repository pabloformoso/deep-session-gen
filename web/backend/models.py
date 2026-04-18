"""Pydantic request / response models."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, EmailStr


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class GenreIntentRequest(BaseModel):
    message: str


class CheckpointRequest(BaseModel):
    feedback: Optional[str] = None


class EditorCommandRequest(BaseModel):
    message: str


class RatingRequest(BaseModel):
    rating: int  # 1–5
    notes: Optional[str] = None
    transition_ratings: Optional[list[dict]] = None
