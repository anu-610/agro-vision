# Agro Vision 🌱 - AI-Powered Plant Disease Diagnosis

**Empowering New Farmers & Home Gardeners with Multimodal AI**

Agro Vision is a smart agricultural assistant designed to make plant disease diagnosis easy and accessible for everyone, especially **new farmers learning the ropes** and **home gardeners** cultivating vegetables for their daily needs. Using a combination of AI technologies, Agro Vision helps you quickly identify potential problems with your plants and get actionable advice.

This project was developed for the **IIT Mandi iHub & HCI Foundation Multimodal AI Hackathon**.

**Live Demo:** [**🚀 Try Agro Vision on Hugging Face Spaces!**](https://huggingface.co/spaces/devil610/agrovision)

## 💡 How It Works: A Multimodal Approach

Agro Vision understands your plant's health through multiple inputs:

1.  **📸 Image Input:** Upload a photo of a plant leaf. Our custom-trained model based on `EfficientNetV2-S` analyzes the image to identify diseases affecting **Potato, Pepper, and Tomato** plants.
2.  **✍️ Text Input:** Describe the symptoms you see or ask a specific question about your plant's health.
3.  **🎤 Voice Input:** Simply click the microphone icon and speak your observations or questions. The browser's built-in speech recognition transcribes your voice into text.
4.  **📍 Location Input (Context):** Optionally share your location. Agro Vision fetches real-time **weather data** (temperature & humidity) for your area. This context helps our AI provide smarter, more relevant diagnoses and solutions (e.g., confirming fungal risk in humid weather).

The backend uses this combined information to query the **Google Gemini API**, generating a context-aware diagnosis and actionable treatment advice.

## 🛠️ Setup & Usage (Local Development)

Follow these steps to run Agro Vision on your local machine.

**Prerequisites:**

* Python 3.10 or higher
* `pip` (Python package installer)
* Git

**Installation:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/anu-610/agro-vision.git
    cd agro-vision
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # For Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

**Running the Application:**

1.  **Start the Flask server:**
    ```bash
    python app.py
    ```
    *(Note: The script is configured to run on `0.0.0.0:5001` for local network access)*
2.  **Open your web browser** and navigate to: `http://localhost:5001` or `http://<your-local-ip>:5001`

## 🚀 Live Deployment

You can access the live version of Agro Vision deployed on Hugging Face Spaces:

[**https://huggingface.co/spaces/devil610/agrovision**](https://huggingface.co/spaces/devil610/agrovision)
