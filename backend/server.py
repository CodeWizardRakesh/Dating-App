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
    age: Optional[int] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    gender: Optional[str] = None
    looking_for: Optional[str] = None
    photos: List[str] = []  # Base64 images
    main_photo_embedding: Optional[List[float]] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)

class Celebrity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    image_url: str
    image_base64: str
    embedding: List[float]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserPreference(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    selected_celebrities: List[str]  # Celebrity IDs
    composite_embedding: List[float]
    age_range: List[int] = [18, 50]
    max_distance: int = 50  # miles
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserAction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    target_user_id: str
    action: str  # "like", "pass", "super_like"
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Match(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user1_id: str
    user2_id: str
    similarity_score: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_message_at: Optional[datetime] = None

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    match_id: str
    sender_id: str
    message: str
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
    return {"message": "FaceMatch Dating API"}

@api_router.post("/auth/register")
async def register_user(
    email: str = Form(...), 
    name: str = Form(...),
    age: int = Form(...),
    bio: str = Form(""),
    location: str = Form(""),
    gender: str = Form(""),
    looking_for: str = Form("")
):
    """Register a new user with profile information"""
    try:
        # Check if user already exists
        existing_user = await db.users.find_one({"email": email})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        user = User(
            email=email,
            name=name,
            age=age,
            bio=bio,
            location=location,
            gender=gender,
            looking_for=looking_for
        )
        await db.users.insert_one(user.dict())
        return {"message": "User registered successfully", "user": user.dict()}
    except HTTPException:
        raise
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
async def set_user_preferences(
    user_id: str, 
    celebrity_ids: List[str],
    age_min: int = Form(18),
    age_max: int = Form(50),
    max_distance: int = Form(50)
):
    """Set user's celebrity preferences and dating filters"""
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
            composite_embedding=composite_embedding,
            age_range=[age_min, age_max],
            max_distance=max_distance
        )
        
        # Remove existing preferences for this user
        await db.user_preferences.delete_many({"user_id": user_id})
        
        # Insert new preferences
        await db.user_preferences.insert_one(preferences.dict())
        
        return {"message": "Preferences saved successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/users/{user_id}/photos")
async def upload_user_photos(user_id: str, files: List[UploadFile] = File(...)):
    """Upload user profile photos"""
    try:
        if len(files) > 6:
            raise HTTPException(status_code=400, detail="Maximum 6 photos allowed")
        
        photos = []
        main_photo_embedding = None
        
        for i, file in enumerate(files):
            # Read and process image
            content = await file.read()
            image = Image.open(io.BytesIO(content))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for consistency
            image = image.resize((400, 400))
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            photos.append(image_base64)
            
            # Use first photo for main embedding
            if i == 0:
                main_photo_embedding = base64_to_embedding(image_base64)
        
        # Update user with photos and main embedding
        await db.users.update_one(
            {"id": user_id},
            {"$set": {
                "photos": photos,
                "main_photo_embedding": main_photo_embedding,
                "last_active": datetime.utcnow()
            }}
        )
        
        return {"message": f"Successfully uploaded {len(photos)} photos", "photo_count": len(photos)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/users/{user_id}/discover")
async def discover_users(user_id: str, limit: int = 10):
    """Discover potential matches based on celebrity preferences"""
    try:
        # Get user preferences
        preferences = await db.user_preferences.find_one({"user_id": user_id})
        if not preferences:
            raise HTTPException(status_code=404, detail="User preferences not found")
        
        # Get current user
        current_user = await db.users.find_one({"id": user_id})
        if not current_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        composite_embedding = preferences["composite_embedding"]
        age_range = preferences["age_range"]
        
        # Get users already interacted with
        interacted_users = await db.user_actions.find({"user_id": user_id}).to_list(1000)
        interacted_user_ids = [action["target_user_id"] for action in interacted_users]
        interacted_user_ids.append(user_id)  # Exclude self
        
        # Build filter criteria
        filter_criteria = {
            "id": {"$nin": interacted_user_ids},
            "is_active": True,
            "photos": {"$ne": []},
            "main_photo_embedding": {"$exists": True, "$ne": None}
        }
        
        # Add age filter if available
        if age_range and current_user.get("age"):
            filter_criteria["age"] = {"$gte": age_range[0], "$lte": age_range[1]}
        
        # Add gender preference filter
        if current_user.get("looking_for") and current_user["looking_for"] != "everyone":
            filter_criteria["gender"] = current_user["looking_for"]
        
        # Get potential matches
        potential_matches = await db.users.find(filter_criteria).to_list(100)
        
        # Calculate similarity scores and rank
        matches_with_scores = []
        for user in potential_matches:
            if user["main_photo_embedding"]:
                similarity = calculate_similarity(composite_embedding, user["main_photo_embedding"])
                matches_with_scores.append({
                    "user": user,
                    "similarity_score": similarity
                })
        
        # Sort by similarity score (descending) and limit results
        matches_with_scores.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_matches = matches_with_scores[:limit]
        
        # Format response
        discovered_users = []
        for match in top_matches:
            user_data = match["user"]
            discovered_users.append({
                "id": user_data["id"],
                "name": user_data["name"],
                "age": user_data.get("age"),
                "bio": user_data.get("bio", ""),
                "location": user_data.get("location", ""),
                "photos": user_data["photos"],
                "similarity_score": match["similarity_score"],
                "match_percentage": int(match["similarity_score"] * 100)
            })
        
        return {
            "users": discovered_users,
            "count": len(discovered_users)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/users/{user_id}/action")
async def user_action(
    user_id: str,
    target_user_id: str = Form(...),
    action: str = Form(...)  # "like", "pass", "super_like"
):
    """Record user action (like, pass, super_like)"""
    try:
        # Check if action already exists
        existing_action = await db.user_actions.find_one({
            "user_id": user_id,
            "target_user_id": target_user_id
        })
        
        if existing_action:
            return {"message": "Action already recorded"}
        
        # Record the action
        user_action = UserAction(
            user_id=user_id,
            target_user_id=target_user_id,
            action=action
        )
        await db.user_actions.insert_one(user_action.dict())
        
        # Check for mutual like to create match
        if action == "like" or action == "super_like":
            mutual_action = await db.user_actions.find_one({
                "user_id": target_user_id,
                "target_user_id": user_id,
                "action": {"$in": ["like", "super_like"]}
            })
            
            if mutual_action:
                # Get user preferences to calculate similarity score
                preferences = await db.user_preferences.find_one({"user_id": user_id})
                target_user = await db.users.find_one({"id": target_user_id})
                
                similarity_score = 0.0
                if preferences and target_user and target_user.get("main_photo_embedding"):
                    similarity_score = calculate_similarity(
                        preferences["composite_embedding"],
                        target_user["main_photo_embedding"]
                    )
                
                # Create match
                match = Match(
                    user1_id=user_id,
                    user2_id=target_user_id,
                    similarity_score=similarity_score
                )
                await db.matches.insert_one(match.dict())
                
                return {
                    "message": "It's a match!",
                    "match_created": True,
                    "match_id": match.id,
                    "similarity_score": similarity_score
                }
        
        return {"message": "Action recorded", "match_created": False}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/users/{user_id}/matches")
async def get_user_matches(user_id: str):
    """Get user's matches"""
    try:
        # Find matches where user is involved
        matches = await db.matches.find({
            "$or": [
                {"user1_id": user_id},
                {"user2_id": user_id}
            ]
        }).to_list(100)
        
        match_data = []
        for match in matches:
            # Get the other user's info
            other_user_id = match["user2_id"] if match["user1_id"] == user_id else match["user1_id"]
            other_user = await db.users.find_one({"id": other_user_id})
            
            if other_user:
                # Get latest message if any
                latest_message = await db.messages.find_one(
                    {"match_id": match["id"]},
                    sort=[("created_at", -1)]
                )
                
                match_data.append({
                    "match_id": match["id"],
                    "user": {
                        "id": other_user["id"],
                        "name": other_user["name"],
                        "age": other_user.get("age"),
                        "photos": other_user["photos"]
                    },
                    "similarity_score": match["similarity_score"],
                    "match_percentage": int(match["similarity_score"] * 100),
                    "created_at": match["created_at"],
                    "latest_message": latest_message["message"] if latest_message else None,
                    "latest_message_at": latest_message["created_at"] if latest_message else None
                })
        
        # Sort by latest activity
        match_data.sort(key=lambda x: x["latest_message_at"] or x["created_at"], reverse=True)
        
        return {"matches": match_data, "count": len(match_data)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/matches/{match_id}/message")
async def send_message(
    match_id: str,
    sender_id: str = Form(...),
    message: str = Form(...)
):
    """Send a message in a match"""
    try:
        # Verify match exists and user is part of it
        match = await db.matches.find_one({"id": match_id})
        if not match:
            raise HTTPException(status_code=404, detail="Match not found")
        
        if sender_id not in [match["user1_id"], match["user2_id"]]:
            raise HTTPException(status_code=403, detail="Not authorized for this match")
        
        # Create message
        new_message = Message(
            match_id=match_id,
            sender_id=sender_id,
            message=message
        )
        await db.messages.insert_one(new_message.dict())
        
        # Update match last_message_at
        await db.matches.update_one(
            {"id": match_id},
            {"$set": {"last_message_at": datetime.utcnow()}}
        )
        
        return {"message": "Message sent successfully", "message_id": new_message.id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/matches/{match_id}/messages")
async def get_match_messages(match_id: str, user_id: str):
    """Get messages for a match"""
    try:
        # Verify user is part of the match
        match = await db.matches.find_one({"id": match_id})
        if not match:
            raise HTTPException(status_code=404, detail="Match not found")
        
        if user_id not in [match["user1_id"], match["user2_id"]]:
            raise HTTPException(status_code=403, detail="Not authorized for this match")
        
        # Get messages
        messages = await db.messages.find(
            {"match_id": match_id}
        ).sort("created_at", 1).to_list(1000)
        
        return {"messages": messages}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/users/{user_id}/profile")
async def get_user_profile(user_id: str):
    """Get user profile"""
    try:
        user = await db.users.find_one({"id": user_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return User(**user)
        
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