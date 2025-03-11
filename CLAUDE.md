# CFD Exploration Project Guidelines

## Build/Run Commands
- Run app: `streamlit run app.py`
- Install dependencies: `pip install -r requirements.txt`
- Deploy to Streamlit Cloud: Use the Streamlit Cloud dashboard

## Code Style Guidelines
- **Imports**: Group imports by functionality (standard lib, visualization, data processing, ML)
- **Functions**: Use descriptive snake_case names with docstrings
- **Error Handling**: Use try/except blocks for critical sections
- **Performance**: Use Streamlit's @st.cache_data for data loading/processing
- **Variables**: Use snake_case for variable names
- **UI Organization**: Use tabs for different analysis views
- **Visualization**: Use Plotly for interactive visualizations
- **Documentation**: Include explanations for complex visualizations in expanders
- **Data Processing**: Standardize data using scikit-learn's StandardScaler
- **Clustering**: Use sklearn clustering algorithms with consistent parameters
- **Formatting**: Follow PEP 8 guidelines for Python code organization

## Project Structure
- Main application: app.py 
- Dependencies: requirements.txt
- Data file: final_states_clean.parquet

## Journal Publication Guidelines

### Figure Requirements
- **File Format**:
  - Vector drawings: EPS or PDF (preferred for line art)
  - Photos/halftones: TIFF, JPG or PNG (min 300 dpi)
  - Line drawings: TIFF, JPG or PNG (min 1000 dpi)
  - Combinations: TIFF, JPG or PNG (min 500 dpi)

- **Resolution Requirements**:
  - Single column width: min 1063 pixels (photos), 3543 pixels (line drawings), 1772 pixels (combinations)
  - Full page width: min 2244 pixels (photos), 7480 pixels (line drawings), 3740 pixels (combinations)

- **Figure Organization**:
  - Submit each figure as separate file
  - Number figures sequentially (Figure_1, Figure_2, etc.)
  - Cite all figures in manuscript text
  - Include captions in separate file

- **Color Guidelines**:
  - Ensure accessibility for those with impaired color vision
  - Consider how figures will appear in print vs. online
  - Use colorblind-friendly palettes (avoid red-green combinations)

- **Best Practices**:
  - Avoid low-resolution screen-optimized formats (GIF, BMP, PICT, WPG)
  - Keep text consistent with manuscript font size
  - Minimize text within figures
  - Explain symbols and abbreviations in caption