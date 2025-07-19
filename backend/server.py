from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import base64
import io
from PIL import Image
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import json
import asyncio
import requests
from serpapi import GoogleSearch

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# SerpAPI key
SERPAPI_KEY = "e0cd4ba49cc494ee6c1eab1bcf0b9bec372b167c6d578ba352d9d1484600a825"

# Define Models
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Celebrity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    image_url: str
    image_base64: str
    embedding: List[float]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    image_base64: str
    embedding: List[float]
    similarity_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class MatchResult(BaseModel):
    profile: UserProfile
    similarity_score: float
    rank: int

class UserPreference(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    selected_celebrities: List[str]  # Celebrity IDs
    composite_embedding: List[float]
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Utility Functions
def image_to_base64(image_path_or_url):
    """Convert image to base64 string"""
    try:
        if image_path_or_url.startswith('http'):
            response = requests.get(image_path_or_url, timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
        else:
            image = Image.open(image_path_or_url)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize image for consistency
        image = image.resize((224, 224))
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def base64_to_embedding(base64_str):
    """Extract facial embedding from base64 image using DeepFace"""
    try:
        # Decode base64 to image
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Extract embedding using DeepFace with FaceNet
        embedding = DeepFace.represent(
            img_array, 
            model_name='Facenet',
            enforce_detection=False  # Don't fail if no face detected
        )
        
        # DeepFace returns a list of embeddings, take the first one
        if isinstance(embedding, list) and len(embedding) > 0:
            return embedding[0]['embedding']
        else:
            return embedding['embedding']
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting embedding: {str(e)}")

def calculate_composite_embedding(embeddings):
    """Calculate composite embedding from multiple celebrity embeddings"""
    if not embeddings:
        return None
    
    # Convert to numpy array and calculate mean
    embeddings_array = np.array(embeddings)
    composite = np.mean(embeddings_array, axis=0)
    return composite.tolist()

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    try:
        # Reshape embeddings for sklearn
        emb1 = np.array(embedding1).reshape(1, -1)
        emb2 = np.array(embedding2).reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

async def search_celebrity_images(celebrity_name, num_images=3):
    """Search for celebrity images using SerpAPI"""
    try:
        params = {
            "q": f"{celebrity_name} face portrait",
            "tbm": "isch",
            "api_key": SERPAPI_KEY,
            "num": num_images
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        images = []
        if "images_results" in results:
            for img in results["images_results"][:num_images]:
                if "original" in img:
                    images.append(img["original"])
                    
        return images
    except Exception as e:
        print(f"Error searching images: {e}")
        return []

# API Routes

@api_router.get("/")
async def root():
    return {"message": "Celebrity Face Matching API"}

@api_router.post("/users/register")
async def register_user(email: str = Form(...), name: str = Form(...)):
    """Register a new user"""
    try:
        user = User(email=email, name=name)
        await db.users.insert_one(user.dict())
        return {"message": "User registered successfully", "user_id": user.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/celebrities")
async def get_celebrities():
    """Get all available celebrities"""
    try:
        celebrities = await db.celebrities.find().to_list(100)
        return [Celebrity(**celeb) for celeb in celebrities]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/celebrities/add")
async def add_celebrity(name: str = Form(...)):
    """Add a new celebrity by searching and processing their images"""
    try:
        # Search for celebrity images
        image_urls = await search_celebrity_images(name, num_images=1)
        
        if not image_urls:
            raise HTTPException(status_code=404, detail="No images found for this celebrity")
        
        # Process the first image
        image_url = image_urls[0]
        image_base64 = image_to_base64(image_url)
        embedding = base64_to_embedding(image_base64)
        
        # Create celebrity record
        celebrity = Celebrity(
            name=name,
            image_url=image_url,
            image_base64=image_base64,
            embedding=embedding
        )
        
        await db.celebrities.insert_one(celebrity.dict())
        return {"message": f"Celebrity {name} added successfully", "celebrity_id": celebrity.id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/users/{user_id}/preferences")
async def set_user_preferences(user_id: str, celebrity_ids: List[str]):
    """Set user's celebrity preferences and calculate composite embedding"""
    try:
        # Get selected celebrities and their embeddings
        celebrities = await db.celebrities.find({"id": {"$in": celebrity_ids}}).to_list(100)
        
        if not celebrities:
            raise HTTPException(status_code=404, detail="No celebrities found")
        
        # Extract embeddings
        embeddings = [celeb["embedding"] for celeb in celebrities]
        
        # Calculate composite embedding
        composite_embedding = calculate_composite_embedding(embeddings)
        
        # Save user preferences
        preferences = UserPreference(
            user_id=user_id,
            selected_celebrities=celebrity_ids,
            composite_embedding=composite_embedding
        )
        
        # Remove existing preferences for this user
        await db.user_preferences.delete_many({"user_id": user_id})
        
        # Insert new preferences
        await db.user_preferences.insert_one(preferences.dict())
        
        return {"message": "Preferences saved successfully", "composite_embedding_length": len(composite_embedding)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/users/{user_id}/upload-profiles")
async def upload_user_profiles(user_id: str, files: List[UploadFile] = File(...)):
    """Upload multiple profile images and extract embeddings"""
    try:
        if len(files) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 images allowed")
        
        uploaded_profiles = []
        
        for file in files:
            # Read and process image
            content = await file.read()
            image = Image.open(io.BytesIO(content))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for consistency
            image = image.resize((224, 224))
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Extract embedding
            embedding = base64_to_embedding(image_base64)
            
            # Create profile record
            profile = UserProfile(
                user_id=user_id,
                image_base64=image_base64,
                embedding=embedding
            )
            
            await db.user_profiles.insert_one(profile.dict())
            uploaded_profiles.append(profile)
        
        return {
            "message": f"Successfully uploaded {len(uploaded_profiles)} profiles",
            "profile_count": len(uploaded_profiles)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/users/{user_id}/matches")
async def get_matches(user_id: str):
    """Get ranked matches based on similarity to user's celebrity preferences"""
    try:
        # Get user preferences
        preferences = await db.user_preferences.find_one({"user_id": user_id})
        if not preferences:
            raise HTTPException(status_code=404, detail="User preferences not found")
        
        composite_embedding = preferences["composite_embedding"]
        
        # Get all uploaded profiles (excluding the requesting user)
        profiles = await db.user_profiles.find({"user_id": {"$ne": user_id}}).to_list(1000)
        
        if not profiles:
            return {"message": "No profiles found", "matches": []}
        
        # Calculate similarity scores
        matches = []
        for profile in profiles:
            similarity = calculate_similarity(composite_embedding, profile["embedding"])
            
            match_result = MatchResult(
                profile=UserProfile(**profile),
                similarity_score=similarity,
                rank=0  # Will be set after sorting
            )
            matches.append(match_result)
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Assign ranks
        for i, match in enumerate(matches):
            match.rank = i + 1
        
        return {
            "user_id": user_id,
            "total_matches": len(matches),
            "matches": [match.dict() for match in matches[:50]]  # Return top 50
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/test-face-detection")
async def test_face_detection():
    """Test endpoint to verify DeepFace is working"""
    try:
        # Create a simple test image
        test_image = Image.new('RGB', (224, 224), color='red')
        buffer = io.BytesIO()
        test_image.save(buffer, format='JPEG')
        test_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Try to extract embedding
        embedding = base64_to_embedding(test_base64)
        
        return {
            "message": "DeepFace is working",
            "embedding_length": len(embedding),
            "sample_values": embedding[:5]
        }
    except Exception as e:
        return {"error": str(e), "message": "DeepFace test failed"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()