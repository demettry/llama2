import streamlit as st
from rag_engine import query_plant_state
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Project Genesis",
    page_icon="🌱",
    layout="wide"
)

# --- Sidebar ---
with st.sidebar:
    st.title("🌱 Project Genesis")
    st.markdown("""
    **A Science-Based Plant Growth Forecaster**

    This tool uses a local Retrieval-Augmented Generation (RAG) system
    to predict a plant's future state based on scientific literature.

    1.  Enter the plant's details and conditions.
    2.  The system queries a knowledge base of scientific papers.
    3.  A local LLM generates a forecast based on the retrieved data.
    """)
    st.warning("⚠️ **Ollama must be running locally** for the prediction to work.")
    st.info("Ensure you have run `python setup.py` to build the knowledge base from the PDFs in the `/papers` directory.")


# --- Main Application ---
st.title("🌿 Plant Growth Forecaster")

# --- Two-Column Layout ---
col1, col2 = st.columns(2)

# --- Column 1: Inputs ---
with col1:
    st.subheader("Input Parameters")

    plant_type = st.text_input(
        "Plant Type",
        placeholder="e.g., Solanum lycopersicum (Tomato)",
        help="Specify the scientific or common name of the plant."
    )

    time_period = st.text_input(
        "Time Period",
        placeholder="e.g., over the next 4 weeks",
        help="Define the duration for the growth prediction."
    )

    conditions = st.text_area(
        "Environmental & 'Strange' Conditions",
        placeholder="e.g., The plant is subjected to a constant temperature of 30°C, high humidity, and is watered with a 2% saline solution.",
        height=150,
        help="Describe the specific environmental factors, treatments, or any unusual conditions the plant will experience."
    )

    baseline_image = st.file_uploader(
        "Upload Baseline Image (Optional)",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of the plant's current state."
    )

    predict_button = st.button("🚀 Predict Growth")

# --- Column 2: Outputs ---
with col2:
    st.subheader("Predicted Future State")

    if predict_button:
        # --- Input Validation ---
        if not all([plant_type, time_period, conditions]):
            st.error("Please fill in all required fields: Plant Type, Time Period, and Conditions.")
        else:
            with st.spinner('Analyzing scientific literature and running prediction... This may take a moment.'):
                # --- Construct the Query ---
                query = (f"Based on scientific principles, what is the predicted state of a "
                         f"'{plant_type}' plant {time_period}, if it is exposed to the following conditions: "
                         f"{conditions}? Provide a detailed botanical description and an image prompt.")

                print(f"Constructed Query: {query}")

                # --- Call the Backend ---
                start_time = time.time()
                result = query_plant_state(query)
                end_time = time.time()

                print(f"Backend processing took {end_time - start_time:.2f} seconds.")

                # --- Display Results ---
                if "Error:" in result["Description"]:
                    st.error(result["Description"])
                else:
                    st.markdown("##### Scientific Description")
                    st.success(result.get("Description", "No description was generated."))

                    st.markdown("---")

                    st.markdown("##### Visualization Prompt")
                    st.info(result.get("Image_Prompt", "No image prompt was generated."))

                    # --- Image Placeholder ---
                    st.image(
                        "https://via.placeholder.com/512x512.png?text=Image+Generation+Result",
                        caption="Placeholder for generated image. The prompt above can be used with an image generation model."
                    )
    else:
        st.info("Enter the plant's conditions on the left and click 'Predict Growth' to see the forecast here.")
