import streamlit as st
import pandas as pd
from io import BytesIO
from pathlib import Path
from prediction.pipeline import predict_file

st.set_page_config(page_title="Prédiction d'épaisseur", layout="centered")
st.title(" Prédiction automatique d'épaisseur")

#  Upload fichiers 
files = st.file_uploader(
    "Fichiers Excel (bruts) :", type=["xls", "xlsx"],
    accept_multiple_files=True
)

#  Consolidation des onglets 
@st.cache_data(show_spinner=False)
def consolidate_excel(file) -> pd.DataFrame:
    xls = pd.ExcelFile(file)
    all_data = []
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet, header=1)
            df.drop(df.columns[0], axis=1, inplace=True)
            df["fichier_source"] = Path(file.name).name
            df["onglet_source"] = sheet
            all_data.append(df)
        except Exception as e:
            st.warning(f"Feuille ignorée ({sheet}) : {e}")
    return pd.concat(all_data, ignore_index=True)

# Affichage des fichiers + onglets via menu déroulant 
db_consolidees = []
fichier_options = {}

if files:
    st.subheader(" Fichiers importés")
    total_onglets = 0

    for f in files:
        try:
            df_full = consolidate_excel(f)
            nb_onglets = df_full["onglet_source"].nunique()
            total_onglets += nb_onglets
            db_consolidees.append(df_full)
            fichier_options[f.name] = (df_full, nb_onglets)
        except Exception as e:
            st.error(f" {f.name} : erreur de lecture - {e}")

    st.caption(f" {len(fichier_options)} fichier(s) chargé(s), {total_onglets} onglet(s) consolidé(s) au total.")

    # Menu déroulant pour sélectionner un fichier
    selected_file = st.selectbox(
        " Sélectionner un fichier pour voir ses onglets :",
        options=list(fichier_options.keys()),
        index=0
    )

    df_selected, nb_selected = fichier_options[selected_file]
    st.markdown(f"### {selected_file} — {nb_selected} onglet(s)")
    st.dataframe(
        df_selected[["onglet_source"]].drop_duplicates().reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

# ——— 4. Choix du modèle ———
model_choice = st.selectbox("Modèle de prédiction :", ["SVR", "RandomForest", "CNN"])

# ——— 5. Lancer la prédiction ———
if st.button("Calculer les prédictions", disabled=not db_consolidees):
    db_finale = pd.concat(db_consolidees, ignore_index=True)
    st.info(" Calcul en cours…")
    try:
        df_pred = predict_file(
            file_like=None,
            model_name=model_choice,
            model=None,
            scaler=None,
            scaler_aux=None,
            df_override=db_finale  
        )

        # Extraction fichier/onglet
        fic_ong = df_pred["onglet_source"].str.split(r"\s*\|\s*", n=1, expand=True)
        df_pred.insert(0, "Fichier", fic_ong[0])
        df_pred.insert(1, "Onglet",  fic_ong[1])
        df_pred.drop(columns="onglet_source", inplace=True)

        # Renommer la colonne
        pred_col = df_pred.columns[-1]
        df_pred.rename(columns={pred_col: "Épaisseur_prévue_mm"}, inplace=True)
        df_pred["Épaisseur_prévue_mm"] = df_pred["Épaisseur_prévue_mm"].round(3)

        st.subheader(" Résultats")
        st.dataframe(df_pred, use_container_width=True)

     
        @st.cache_data
        def to_excel(df: pd.DataFrame) -> bytes:
            out = BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Prédictions", index=False)
                ws = writer.sheets["Prédictions"]
                for col in ws.columns:
                    max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
                    ws.column_dimensions[col[0].column_letter].width = max_len + 2
            return out.getvalue()

        xlsx_bytes = to_excel(df_pred)
        st.download_button(
            "Télécharger Excel (XLSX)",
            data=xlsx_bytes,
            file_name="predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Erreur pendant la prédiction : {e}")
