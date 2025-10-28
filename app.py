import os
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import requests
import google.generativeai as genai

# --- 1. CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDEj6-DQczZxrxLRjH3J0gocb2N7H7PVRs")
genai.configure(api_key=GEMINI_API_KEY)
generation_config = { "temperature": 0.7, "top_p": 1, "top_k": 1, "max_output_tokens": 2048 }
safety_settings = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
gemini_model = genai.GenerativeModel(model_name="gemini-2.5-flash", generation_config=generation_config, safety_settings=safety_settings)

# --- 2. LOAD VISION MODEL & CLASS NAMES ---
# (This section is unchanged - downloads model if it doesn't exist)
MODEL_URL = "https://github.com/anu-610/agro-vision/raw/refs/heads/main/model.h5?download="
MODEL_PATH = "model.h5"
if not os.path.exists(MODEL_PATH) and MODEL_URL != "YOUR_DIRECT_DOWNLOAD_LINK_HERE":
    print(f"Downloading model...")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded!")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Vision model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# --- 3. HELPER FUNCTIONS ---

def predict_disease_from_image(image_path):
    # (This function is unchanged)
    if model is None: return "Error: Vision model not loaded.", 0.0
    try:
        img = Image.open(image_path).resize((224, 224))
        img_array = np.array(img); img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array)
        idx = np.argmax(preds[0]); name = class_names[idx].replace('___', ' - ').replace('_', ' ')
        return name, np.max(preds[0])
    except Exception as e: return f"Error processing image: {e}", 0.0

def get_gemini_response(prompt):
    # (This function is unchanged)
    try:
        convo = gemini_model.start_chat(history=[]); convo.send_message(prompt)
        return convo.last.text
    except Exception as e: return f"Error from AI: {e}"

# --- NEW HELPER FUNCTIONS for LOCATION & WEATHER ---
def get_location_name(lat, lon):
    """Gets a city/state name from coordinates using a free reverse geocoding API."""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
        headers = {'User-Agent': 'CropProtectorApp/1.0'}
        response = requests.get(url, headers=headers).json()
        address = response.get('address', {})
        # Try to find the most relevant location name
        city = address.get('city', address.get('town', address.get('village')))
        state = address.get('state')
        country = address.get('country')
        if city and state: return f"{city}, {state}"
        if state and country: return f"{state}, {country}"
        if country: return country
        return "Unknown Location"
    except Exception as e:
        print(f"Error getting location name: {e}")
        return None

def get_weather_data(lat, lon):
    """Gets current temperature and humidity from a free weather API."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
        response = requests.get(url).json()
        current_weather = response.get('current', {})
        return {
            "temperature": current_weather.get('temperature_2m'),
            "humidity": current_weather.get('relative_humidity_2m')
        }
    except Exception as e:
        print(f"Error getting weather data: {e}")
        return None

# --- MODIFIED MULTIMODAL ANALYSIS ---
def analyze_multimodal_query(image_disease=None, text_query=None, weather_data=None):
    system_prompt = (
        "You are an expert agricultural assistant named Agro Vision. Provide concise, actionable advice for farmers. "
        "**Your key skill is using environmental context.**\n"
        "RULES:\n"
        "1.  If weather data (temperature, humidity) is available, YOU MUST use it to make your diagnosis more accurate. For example, high humidity supports fungal growth. Hot, dry weather might mean sun scorch, not a fungus.\n"
        "2.  If the user's text and the image diagnosis conflict (e.g., text says 'potato', image shows 'tomato'), address both separately.\n"
        "3.  If the text query is not about farming, politely decline and analyze the image if one was provided."
    )
    
    final_prompt = f"{system_prompt}\n\nHere is the situation:\n"
    if image_disease:
        final_prompt += f"- My vision model analyzed an image and identified: **{image_disease}**.\n"
    if text_query:
        final_prompt += f"- The user also provided this text query: \"**{text_query}**\"\n"
    if weather_data and weather_data.get('temperature') is not None:
        final_prompt += f"- The user's current weather is: **Temperature: {weather_data['temperature']}°C, Humidity: {weather_data['humidity']}%**.\n"
    
    final_prompt += "\nPlease provide a complete, context-aware response."
    return get_gemini_response(final_prompt)


# --- 4. FLASK APP & ROUTES (MODIFIED) ---
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None: return jsonify({'error': 'The vision model is not loaded.'}), 500

    image_file = request.files.get('file')
    text_query = request.form.get('text_query', '').strip()
    latitude = request.form.get('latitude')
    longitude = request.form.get('longitude')
    
    result = {"disease": None, "confidence": None, "solution": None, "location_name": None}
    
    if not image_file and not text_query:
        return jsonify({'error': 'Please upload an image or ask a question.'}), 400

    # --- 1. Get Location & Weather ---
    weather_data = None
    if latitude and longitude:
        result["location_name"] = get_location_name(latitude, longitude)
        weather_data = get_weather_data(latitude, longitude)

    # --- 2. Process Image ---
    if image_file:
        try:
            filepath = os.path.join('uploads', image_file.filename); os.makedirs('uploads', exist_ok=True)
            image_file.save(filepath)
            disease_name, confidence = predict_disease_from_image(filepath)
            os.remove(filepath)
            if "Error" in disease_name: return jsonify({'error': disease_name}), 500
            result["disease"] = disease_name
            result["confidence"] = f"{confidence:.2%}"
        except Exception as e: return jsonify({'error': f'Error saving file: {e}'}), 500

    # --- 3. Get Gemini's Context-Aware Response ---
    result["solution"] = analyze_multimodal_query(
        image_disease=result.get("disease"),
        text_query=text_query,
        weather_data=weather_data
    )

    return jsonify(result)

if __name__ == "__main__":
    app.run("0.0.0.0", 5001, debug=True)