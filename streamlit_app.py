import requests
import streamlit as st
import pandas as pd

API_URL = "http://127.0.0.1:8000/extract"

st.set_page_config(page_title="Beluo Document Intelligence", layout="wide")

st.title("Beluo - Extraction de devis")
st.write("Upload d'un devis PDF, extraction des lignes métier, normalisation vers le format Beluo.")

uploaded_file = st.file_uploader("Déposer un devis PDF", type=["pdf"])

if uploaded_file is not None:
    st.subheader("Document chargé")
    st.write(uploaded_file.name)

    if st.button("Extraire les données"):
        with st.spinner("Extraction en cours..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf",
                )
            }

            response = requests.post(API_URL, files=files)

        if response.status_code != 200:
            st.error(f"Erreur API : {response.status_code}")
            st.text(response.text)
        else:
            result = response.json()

            st.subheader("Statut")
            st.write(result.get("status"))

            if result.get("status") == "extracted":
                items = result.get("items", [])

                st.success(f"{len(items)} ligne(s) extraite(s)")

                df = pd.DataFrame(items)

                st.subheader("Tableau normalisé Beluo")
                st.dataframe(df, use_container_width=True)

                st.subheader("JSON Beluo")
                st.json(items)

            else:
                st.warning("Document à revoir manuellement")
                st.write(result.get("reason"))