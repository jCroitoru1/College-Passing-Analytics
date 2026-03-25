# CFB Streamlit App

This app wraps the existing `cfb_multi_model_pipeline.py` workflow and presents the generated artifacts in a Streamlit UI.

## Run

From `C:\Users\jcroi\OneDrive\Documents\CFB DATA`:

```powershell
streamlit run streamlit/app.py
```

## Notes

- The app reads generated files from `outputs/`.
- Use the sidebar button to rerun the analysis and refresh the outputs from the UI.
- If dependencies are missing, install from `streamlit/requirements.txt`.
