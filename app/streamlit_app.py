import requests
import streamlit as st

import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)
st.set_page_config(
    page_title="Document Intelligence",
    layout="centered",
)

st.title("Analyse de devis")

st.write(
    "Déposez un devis PDF pour extraire automatiquement "
    "les informations principales."
)


uploaded_file = st.file_uploader(
    "Déposer un devis PDF",
    type=["pdf"],
)


if uploaded_file is not None:

    st.subheader("Document chargé")
    st.write(uploaded_file.name)

    with st.spinner("Analyse du document en cours..."):

        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf",
                )
            }

            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                timeout=120,
            )

            response.raise_for_status()

            result = response.json()

            st.success("Analyse terminée")

            st.subheader("Informations extraites")

            st.write(
                "**Client :**",
                result.get("client") or "Non détecté",
            )

            st.write(
                "**Fournisseur :**",
                result.get("fournisseur") or "Non détecté",
            )

            st.write(
                "**Numéro du devis :**",
                result.get("numero_devis") or "Non détecté",
            )

            st.write(
                "**Date du devis :**",
                result.get("date_devis") or "Non détecté",
            )

            st.write(
                "**Montant total :**",
                result.get("montant_total") or "Non détecté",
            )

            with st.expander("Afficher le JSON"):
                st.json(result)

        except requests.HTTPError:
            st.error(
                f"Erreur API : {response.status_code}"
            )

            st.code(response.text)

        except requests.ConnectionError:
            st.error(
                "Impossible de contacter l'API FastAPI. "
                "Vérifiez qu'elle est démarrée."
            )

        except requests.Timeout:
            st.error(
                "L'analyse du document a dépassé le délai autorisé."
            )

        except requests.RequestException as error:
            st.error(
                f"Erreur lors de l'appel à l'API : {error}"
            )