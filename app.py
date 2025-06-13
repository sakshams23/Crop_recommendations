import streamlit as st
import pickle
from PIL import Image
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key is not set. Check your .env file or environment variables.")
else:
    genai.configure(api_key=api_key)

# Load ML models
svm_model = pickle.load(open('SVM.pkl', 'rb'))
DecisionTree_model = pickle.load(open('DecisionTree.pkl', 'rb'))
NaiveBayes_model = pickle.load(open('NaiveBayes.pkl', 'rb'))
RF_model = pickle.load(open('RF.pkl', 'rb'))

# Crop classification function
def classify(answer):
    return answer[0] + " is the best crop for cultivation in this condition."

# Gemini API response function
def get_gemini_response(input_text, image_data, prompt):
    if not input_text or not image_data or not prompt:
        raise ValueError("All parameters (input_text, image_data, and prompt) must be non-empty.")

    model = genai.GenerativeModel('gemini-1.5-flash-002')
    response = model.generate_content([input_text, image_data[0], prompt])
    return response.text

# Image processing function
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{
            "mime_type": uploaded_file.type,
            "data": bytes_data
        }]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Main app
def main():
    st.set_page_config(page_title="Crop Recommendation and Plant Analysis")

    st.title("🌱 Smart Agriculture System")
    app_mode = st.sidebar.selectbox("Select Feature", ["Crop Recommendation", "Plant Disease Detection"])

    if app_mode == "Crop Recommendation":
        crop_recommendation()
    elif app_mode == "Plant Disease Detection":
        plant_disease_detection()

# Crop Recommendation App
def crop_recommendation():
    st.header("🌾 Crop Recommendation using ML and IoT")

    activities = [
        'Naive Bayes (Accuracy: 98.86%)',
        'Decision Tree (Accuracy: 90.68%)',
        'SVM (Accuracy: 97.72%)',
        'Random Forest (Accuracy: 99.54%)'
    ]
    option = st.sidebar.selectbox("Choose model?", activities)
    st.subheader(option)

    sn = st.slider('NITROGEN (N)', 0.0, 200.0)
    sp = st.slider('PHOSPHOROUS (P)', 0.0, 200.0)
    pk = st.slider('POTASSIUM (K)', 0.0, 200.0)
    pt = st.slider('TEMPERATURE', 0.0, 50.0)
    phu = st.slider('HUMIDITY', 0.0, 100.0)
    pPh = st.slider('Ph', 0.0, 14.0)
    pr = st.slider('RAINFALL', 0.0, 300.0)

    inputs = [[sn, sp, pk, pt, phu, pPh, pr]]

    if st.button('Predict'):
        if option == 'SVM (Accuracy: 97.72%)':
            st.success(classify(svm_model.predict(inputs)))
        elif option == 'Decision Tree (Accuracy: 90.68%)':
            st.success(classify(DecisionTree_model.predict(inputs)))
        elif option == 'Naive Bayes (Accuracy: 98.86%)':
            st.success(classify(NaiveBayes_model.predict(inputs)))
        else:
            st.success(classify(RF_model.predict(inputs)))

# Plant Disease Detection App
def plant_disease_detection():
    st.header("🪴 Plant Disease Detection using Gemini AI")

    input_text = st.text_input("Input Prompt: ", key="input")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Tell me about the image"):
        if not input_text.strip():
            st.error("Input prompt cannot be empty. Please provide a valid text input.")
        else:
            try:
                image_data = input_image_setup(uploaded_file)

                input_prompt = """
                    You are a Plant expert. Analyze the plant in the image and provide the following:
                    Plant Biological Name:...
                    Issue:...
                    Care Needed:...
                    Conclusion:...
                """

                response = get_gemini_response(input_text, image_data, input_prompt)
                st.subheader("The Response is:")
                st.write(response)

            except FileNotFoundError:
                st.error("Please upload an image.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
