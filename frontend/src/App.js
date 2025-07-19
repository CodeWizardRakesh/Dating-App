import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function App() {
  const [currentStep, setCurrentStep] = useState('register');
  const [user, setUser] = useState(null);
  const [celebrities, setCelebrities] = useState([]);
  const [selectedCelebrities, setSelectedCelebrities] = useState([]);
  const [discoveredUsers, setDiscoveredUsers] = useState([]);
  const [currentUserIndex, setCurrentUserIndex] = useState(0);
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showMatches, setShowMatches] = useState(false);

  // Load celebrities on component mount
  useEffect(() => {
    loadCelebrities();
  }, []);

  const loadCelebrities = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/celebrities`);
      setCelebrities(response.data);
    } catch (error) {
      console.error('Error loading celebrities:', error);
      setError('Failed to load celebrities');
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    const formData = new FormData(e.target);
    
    try {
      const response = await axios.post(`${API_URL}/api/auth/register`, formData);
      setUser(response.data.user);
      setCurrentStep('celebrities');
    } catch (error) {
      setError('Registration failed: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleAddCelebrity = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    const formData = new FormData(e.target);
    
    try {
      await axios.post(`${API_URL}/api/celebrities/add`, formData);
      await loadCelebrities();
      e.target.reset();
    } catch (error) {
      setError('Failed to add celebrity: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const toggleCelebritySelection = (celebrityId) => {
    setSelectedCelebrities(prev => 
      prev.includes(celebrityId) 
        ? prev.filter(id => id !== celebrityId)
        : [...prev, celebrityId]
    );
  };

  const handleSavePreferences = async () => {
    if (selectedCelebrities.length === 0) {
      setError('Please select at least one celebrity');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      selectedCelebrities.forEach(id => formData.append('celebrity_ids', id));
      formData.append('age_min', '18');
      formData.append('age_max', '50');
      formData.append('max_distance', '50');

      await axios.post(`${API_URL}/api/users/${user.id}/preferences`, formData);
      setCurrentStep('photos');
    } catch (error) {
      setError('Failed to save preferences: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handlePhotoUpload = async (e) => {
    const files = Array.from(e.target.files);
    
    if (files.length > 6) {
      setError('Maximum 6 photos allowed');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      files.forEach(file => formData.append('files', file));

      await axios.post(`${API_URL}/api/users/${user.id}/photos`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setCurrentStep('discover');
      await loadDiscoverUsers();
    } catch (error) {
      setError('Upload failed: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const loadDiscoverUsers = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_URL}/api/users/${user.id}/discover?limit=20`);
      setDiscoveredUsers(response.data.users || []);
      setCurrentUserIndex(0);
    } catch (error) {
      setError('Failed to load users: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleUserAction = async (action) => {
    if (currentUserIndex >= discoveredUsers.length) return;

    const currentDiscoveredUser = discoveredUsers[currentUserIndex];
    
    try {
      const formData = new FormData();
      formData.append('target_user_id', currentDiscoveredUser.id);
      formData.append('action', action);

      const response = await axios.post(`${API_URL}/api/users/${user.id}/action`, formData);
      
      if (response.data.match_created) {
        // Show match notification
        alert(`🎉 It's a Match! ${response.data.similarity_score ? `${Math.round(response.data.similarity_score * 100)}% similarity` : ''}`);
        await loadMatches();
      }
      
      // Move to next user
      setCurrentUserIndex(prev => prev + 1);
      
      // Load more users if we're running low
      if (currentUserIndex >= discoveredUsers.length - 3) {
        await loadDiscoverUsers();
      }
      
    } catch (error) {
      setError('Action failed: ' + (error.response?.data?.detail || error.message));
    }
  };

  const loadMatches = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/users/${user.id}/matches`);
      setMatches(response.data.matches || []);
    } catch (error) {
      console.error('Failed to load matches:', error);
    }
  };

  const renderRegisterStep = () => (
    <div className="max-w-md mx-auto bg-white rounded-xl shadow-lg p-8">
      <h2 className="text-3xl font-bold text-gray-900 mb-6 text-center">Join FaceMatch</h2>
      <p className="text-gray-600 mb-6 text-center">
        Find your perfect match based on celebrity facial similarity
      </p>
      
      <form onSubmit={handleRegister} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
          <input
            type="text"
            name="name"
            required
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            placeholder="Enter your name"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
          <input
            type="email"
            name="email"
            required
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            placeholder="Enter your email"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Age</label>
          <input
            type="number"
            name="age"
            required
            min="18"
            max="100"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            placeholder="Your age"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Bio</label>
          <textarea
            name="bio"
            rows="3"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            placeholder="Tell us about yourself..."
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
          <input
            type="text"
            name="location"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            placeholder="City, State"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Gender</label>
          <select
            name="gender"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            <option value="">Select gender</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="non-binary">Non-binary</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Looking for</label>
          <select
            name="looking_for"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            <option value="">Select preference</option>
            <option value="male">Men</option>
            <option value="female">Women</option>
            <option value="everyone">Everyone</option>
          </select>
        </div>
        
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white py-3 rounded-lg font-medium hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 transition-all duration-200"
        >
          {loading ? 'Creating Account...' : 'Get Started'}
        </button>
      </form>
    </div>
  );

  const renderCelebritiesStep = () => (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Choose Your Celebrity Preferences</h2>
        <p className="text-gray-600">Select celebrities whose facial features you find attractive</p>
      </div>

      {/* Add Celebrity Form */}
      <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
        <h3 className="text-xl font-semibold mb-4">Add New Celebrity</h3>
        <form onSubmit={handleAddCelebrity} className="flex gap-4">
          <input
            type="text"
            name="name"
            placeholder="Celebrity name (e.g., Emma Stone)"
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
          <button
            type="submit"
            disabled={loading}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Adding...' : 'Add Celebrity'}
          </button>
        </form>
      </div>

      {/* Celebrity Grid */}
      {celebrities.length > 0 ? (
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {celebrities.map((celebrity) => (
              <div
                key={celebrity.id}
                onClick={() => toggleCelebritySelection(celebrity.id)}
                className={`relative cursor-pointer rounded-lg overflow-hidden border-4 transition-all duration-200 ${
                  selectedCelebrities.includes(celebrity.id)
                    ? 'border-purple-500 ring-4 ring-purple-200'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className="aspect-square bg-gray-100 flex items-center justify-center">
                  {celebrity.image_base64 ? (
                    <img
                      src={`data:image/jpeg;base64,${celebrity.image_base64}`}
                      alt={celebrity.name}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="text-gray-400 text-center">
                      <div className="text-4xl mb-2">👤</div>
                      <div className="text-sm">{celebrity.name}</div>
                    </div>
                  )}
                </div>
                <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-75 text-white p-2 text-center text-sm font-medium">
                  {celebrity.name}
                </div>
                {selectedCelebrities.includes(celebrity.id) && (
                  <div className="absolute top-2 right-2 bg-purple-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">
                    ✓
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="bg-white rounded-xl shadow-lg p-8 text-center text-gray-500 mb-8">
          No celebrities available. Add some celebrities to get started!
        </div>
      )}

      {selectedCelebrities.length > 0 && (
        <div className="text-center">
          <p className="text-gray-600 mb-4">
            Selected {selectedCelebrities.length} celebrity{selectedCelebrities.length !== 1 ? 'ies' : 'y'}
          </p>
          <button
            onClick={handleSavePreferences}
            disabled={loading}
            className="bg-gradient-to-r from-purple-600 to-pink-600 text-white px-8 py-3 rounded-lg font-medium hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 transition-all duration-200"
          >
            {loading ? 'Saving...' : 'Continue to Photos'}
          </button>
        </div>
      )}
    </div>
  );

  const renderPhotosStep = () => (
    <div className="max-w-2xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Upload Your Photos</h2>
        <p className="text-gray-600">Add up to 6 photos to create your dating profile</p>
      </div>

      <div className="bg-white rounded-xl shadow-lg p-8">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-purple-400 transition-colors duration-200">
          <div className="text-6xl text-gray-400 mb-4">📸</div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">Choose Your Best Photos</h3>
          <p className="text-gray-600 mb-6">Select up to 6 photos (JPG, PNG) - your first photo will be your main profile picture</p>
          
          <input
            type="file"
            multiple
            accept="image/*"
            onChange={handlePhotoUpload}
            className="hidden"
            id="photo-upload"
          />
          <label
            htmlFor="photo-upload"
            className="inline-block bg-gradient-to-r from-purple-600 to-pink-600 text-white px-8 py-3 rounded-lg font-medium hover:from-purple-700 hover:to-pink-700 cursor-pointer transition-all duration-200"
          >
            {loading ? 'Processing Photos...' : 'Select Photos'}
          </label>
        </div>
      </div>
    </div>
  );

  const renderDiscoverStep = () => {
    const currentDiscoveredUser = discoveredUsers[currentUserIndex];
    
    if (!currentDiscoveredUser) {
      return (
        <div className="max-w-md mx-auto text-center py-12">
          <div className="text-6xl text-gray-400 mb-4">💔</div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">No more users</h2>
          <p className="text-gray-600 mb-6">Check back later for new matches!</p>
          <button
            onClick={loadDiscoverUsers}
            className="bg-purple-600 text-white px-6 py-2 rounded-lg hover:bg-purple-700"
          >
            Refresh
          </button>
        </div>
      );
    }

    return (
      <div className="max-w-md mx-auto">
        <div className="bg-white rounded-xl shadow-lg overflow-hidden">
          {/* Main Photo */}
          <div className="h-96 bg-gray-100 relative">
            <img
              src={`data:image/jpeg;base64,${currentDiscoveredUser.photos[0]}`}
              alt={currentDiscoveredUser.name}
              className="w-full h-full object-cover"
            />
            <div className="absolute top-4 right-4 bg-gradient-to-r from-purple-100 to-pink-100 text-purple-800 px-3 py-1 rounded-full text-sm font-medium">
              {currentDiscoveredUser.match_percentage}% match
            </div>
          </div>

          {/* User Info */}
          <div className="p-6">
            <h3 className="text-2xl font-bold text-gray-900 mb-2">
              {currentDiscoveredUser.name}
              {currentDiscoveredUser.age && (
                <span className="text-gray-600 font-normal">, {currentDiscoveredUser.age}</span>
              )}
            </h3>
            
            {currentDiscoveredUser.location && (
              <p className="text-gray-600 mb-2">📍 {currentDiscoveredUser.location}</p>
            )}
            
            {currentDiscoveredUser.bio && (
              <p className="text-gray-700 mb-4">{currentDiscoveredUser.bio}</p>
            )}

            {/* Photo Gallery */}
            {currentDiscoveredUser.photos.length > 1 && (
              <div className="mb-4">
                <div className="flex space-x-2 overflow-x-auto">
                  {currentDiscoveredUser.photos.slice(1).map((photo, index) => (
                    <img
                      key={index}
                      src={`data:image/jpeg;base64,${photo}`}
                      alt={`${currentDiscoveredUser.name} ${index + 2}`}
                      className="w-16 h-16 object-cover rounded-lg flex-shrink-0"
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex space-x-4">
              <button
                onClick={() => handleUserAction('pass')}
                className="flex-1 bg-gray-200 text-gray-700 py-3 rounded-lg font-medium hover:bg-gray-300 transition-colors duration-200"
              >
                ❌ Pass
              </button>
              <button
                onClick={() => handleUserAction('super_like')}
                className="bg-blue-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-600 transition-colors duration-200"
              >
                ⭐ Super Like
              </button>
              <button
                onClick={() => handleUserAction('like')}
                className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 text-white py-3 rounded-lg font-medium hover:from-purple-700 hover:to-pink-700 transition-all duration-200"
              >
                💜 Like
              </button>
            </div>
          </div>
        </div>

        {/* Progress Indicator */}
        <div className="text-center mt-4 text-gray-500 text-sm">
          User {currentUserIndex + 1} of {discoveredUsers.length}
        </div>
      </div>
    );
  };

  const renderMatches = () => (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Your Matches</h2>
        <p className="text-gray-600">People who liked you back</p>
      </div>

      {matches.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {matches.map((match) => (
            <div key={match.match_id} className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow duration-200">
              <div className="aspect-square bg-gray-100">
                <img
                  src={`data:image/jpeg;base64,${match.user.photos[0]}`}
                  alt={match.user.name}
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="p-4">
                <h3 className="text-lg font-bold text-gray-900 mb-1">{match.user.name}</h3>
                <p className="text-gray-600 text-sm mb-2">{match.match_percentage}% celebrity similarity match</p>
                {match.latest_message && (
                  <p className="text-gray-700 text-sm mb-2 italic">"{match.latest_message}"</p>
                )}
                <button className="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white py-2 rounded-lg font-medium hover:from-purple-700 hover:to-pink-700 transition-all duration-200">
                  Send Message
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <div className="text-6xl text-gray-400 mb-4">💔</div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">No matches yet</h3>
          <p className="text-gray-600">Keep swiping to find your perfect match!</p>
        </div>
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
              FaceMatch
            </h1>
            {user && currentStep === 'discover' && (
              <div className="flex items-center space-x-4">
                <span className="text-gray-600">Hi, {user.name}!</span>
                <button
                  onClick={() => {
                    setShowMatches(!showMatches);
                    if (!showMatches) loadMatches();
                  }}
                  className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition-colors duration-200"
                >
                  {showMatches ? 'Discover' : `Matches ${matches.length > 0 ? `(${matches.length})` : ''}`}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg">
            {error}
          </div>
        )}

        {currentStep === 'register' && renderRegisterStep()}
        {currentStep === 'celebrities' && renderCelebritiesStep()}
        {currentStep === 'photos' && renderPhotosStep()}
        {currentStep === 'discover' && !showMatches && renderDiscoverStep()}
        {currentStep === 'discover' && showMatches && renderMatches()}
      </div>
    </div>
  );
}

export default App;