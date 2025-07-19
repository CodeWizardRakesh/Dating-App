import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function App() {
  const [currentStep, setCurrentStep] = useState('register');
  const [user, setUser] = useState(null);
  const [celebrities, setCelebrities] = useState([]);
  const [selectedCelebrities, setSelectedCelebrities] = useState([]);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

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
      const response = await axios.post(`${API_URL}/api/users/register`, formData);
      setUser({ id: response.data.user_id, ...Object.fromEntries(formData) });
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
      await axios.post(
        `${API_URL}/api/users/${user.id}/preferences`, 
        selectedCelebrities,
        { headers: { 'Content-Type': 'application/json' } }
      );
      setCurrentStep('upload');
    } catch (error) {
      setError('Failed to save preferences: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    
    if (files.length > 100) {
      setError('Maximum 100 images allowed');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      files.forEach(file => formData.append('files', file));

      await axios.post(`${API_URL}/api/users/${user.id}/upload-profiles`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setUploadedFiles(files);
      setCurrentStep('results');
      
      // Load matches
      await loadMatches();
    } catch (error) {
      setError('Upload failed: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const loadMatches = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_URL}/api/users/${user.id}/matches`);
      setMatches(response.data.matches || []);
    } catch (error) {
      setError('Failed to load matches: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const renderRegisterStep = () => (
    <div className="max-w-md mx-auto bg-white rounded-xl shadow-lg p-8">
      <h2 className="text-3xl font-bold text-gray-900 mb-6 text-center">Join FaceMatch</h2>
      <p className="text-gray-600 mb-6 text-center">
        Find profiles similar to your favorite celebrities
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
            {loading ? 'Saving...' : 'Continue to Upload'}
          </button>
        </div>
      )}
    </div>
  );

  const renderUploadStep = () => (
    <div className="max-w-2xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Upload Profile Images</h2>
        <p className="text-gray-600">Upload up to 100 profile images to find the best matches</p>
      </div>

      <div className="bg-white rounded-xl shadow-lg p-8">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-purple-400 transition-colors duration-200">
          <div className="text-6xl text-gray-400 mb-4">📸</div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">Choose Profile Images</h3>
          <p className="text-gray-600 mb-6">Select multiple images (JPG, PNG) - up to 100 files</p>
          
          <input
            type="file"
            multiple
            accept="image/*"
            onChange={handleFileUpload}
            className="hidden"
            id="file-upload"
          />
          <label
            htmlFor="file-upload"
            className="inline-block bg-gradient-to-r from-purple-600 to-pink-600 text-white px-8 py-3 rounded-lg font-medium hover:from-purple-700 hover:to-pink-700 cursor-pointer transition-all duration-200"
          >
            {loading ? 'Processing Images...' : 'Select Images'}
          </label>
        </div>

        {uploadedFiles.length > 0 && (
          <div className="mt-6 p-4 bg-green-50 rounded-lg">
            <p className="text-green-800 font-medium">
              ✅ Successfully uploaded {uploadedFiles.length} images
            </p>
          </div>
        )}
      </div>
    </div>
  );

  const renderResultsStep = () => (
    <div className="max-w-6xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Your Matches</h2>
        <p className="text-gray-600">
          Your images ranked by similarity to selected celebrities
        </p>
        {matches.length > 0 && (
          <div className="text-sm text-gray-500 mt-2">
            <p>Analyzed {matches.length} of your images</p>
            {matches[0]?.all_celebrity_scores && (
              <p>Compared against: {matches[0].all_celebrity_scores.map(s => s.celebrity_name).join(', ')}</p>
            )}
          </div>
        )}
      </div>

      {loading ? (
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600"></div>
          <p className="text-gray-600 mt-4">Loading matches...</p>
        </div>
      ) : matches.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {matches.map((match, index) => (
            <div key={index} className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow duration-200">
              <div className="aspect-square bg-gray-100">
                <img
                  src={`data:image/jpeg;base64,${match.profile.image_base64}`}
                  alt={`Your Image ${index + 1}`}
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="p-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-lg font-bold text-purple-600">#{match.rank}</span>
                  <span className="bg-gradient-to-r from-purple-100 to-pink-100 text-purple-800 px-3 py-1 rounded-full text-sm font-medium">
                    {(match.similarity_score * 100).toFixed(1)}% match
                  </span>
                </div>
                <div className="mb-2">
                  <p className="text-sm text-gray-600">Best match: <span className="font-semibold text-gray-800">{match.best_celebrity_match}</span></p>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mb-3">
                  <div
                    className="bg-gradient-to-r from-purple-600 to-pink-600 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${match.similarity_score * 100}%` }}
                  ></div>
                </div>
                {match.all_celebrity_scores && (
                  <div className="text-xs text-gray-500">
                    <p className="font-medium mb-1">All celebrity similarities:</p>
                    {match.all_celebrity_scores.map((score, idx) => (
                      <div key={idx} className="flex justify-between">
                        <span>{score.celebrity_name}:</span>
                        <span>{(score.similarity_score * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <div className="text-6xl text-gray-400 mb-4">🔍</div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">No matches found</h3>
          <p className="text-gray-600">Try uploading more profile images or adjusting your celebrity preferences</p>
          <button
            onClick={() => setCurrentStep('celebrities')}
            className="mt-4 bg-purple-600 text-white px-6 py-2 rounded-lg hover:bg-purple-700 transition-colors duration-200"
          >
            Adjust Preferences
          </button>
        </div>
      )}

      {matches.length > 0 && (
        <div className="text-center mt-8">
          <button
            onClick={() => setCurrentStep('celebrities')}
            className="bg-gray-600 text-white px-6 py-2 rounded-lg hover:bg-gray-700 mr-4 transition-colors duration-200"
          >
            Change Preferences
          </button>
          <button
            onClick={() => setCurrentStep('upload')}
            className="bg-purple-600 text-white px-6 py-2 rounded-lg hover:bg-purple-700 transition-colors duration-200"
          >
            Upload More Images
          </button>
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
            {user && (
              <div className="flex items-center space-x-4">
                <span className="text-gray-600">Welcome, {user.name}</span>
                <div className="flex space-x-2">
                  <div className={`w-3 h-3 rounded-full ${currentStep === 'register' ? 'bg-purple-600' : currentStep === 'celebrities' || currentStep === 'upload' || currentStep === 'results' ? 'bg-green-500' : 'bg-gray-300'}`}></div>
                  <div className={`w-3 h-3 rounded-full ${currentStep === 'celebrities' ? 'bg-purple-600' : currentStep === 'upload' || currentStep === 'results' ? 'bg-green-500' : 'bg-gray-300'}`}></div>
                  <div className={`w-3 h-3 rounded-full ${currentStep === 'upload' ? 'bg-purple-600' : currentStep === 'results' ? 'bg-green-500' : 'bg-gray-300'}`}></div>
                  <div className={`w-3 h-3 rounded-full ${currentStep === 'results' ? 'bg-purple-600' : 'bg-gray-300'}`}></div>
                </div>
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
        {currentStep === 'upload' && renderUploadStep()}
        {currentStep === 'results' && renderResultsStep()}
      </div>
    </div>
  );
}

export default App;