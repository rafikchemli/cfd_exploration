# Makefile for CFD figure generation

.PHONY: all clean figures

# Default target
all: figures

# Configuration
OUTPUT_DIR = figures
DATA_FILE = final_states_clean.parquet
ALGORITHM = Spectral
CLUSTERS = 8
FEATURES = primary_flow pressure density particle_feed diameter mass
LABELS = "Primary gas flow (kg/s)" "Pressure (bar)" "Particle density (kg/m3)" "Particle feed rate (kg/s)" "diameter" "mass"

# Create figures
figures:
	@echo "Generating publication-quality figures for CFD analysis..."
	@mkdir -p $(OUTPUT_DIR)
	@python3 figure_export.py \
		--data $(DATA_FILE) \
		--output $(OUTPUT_DIR) \
		--n_clusters $(CLUSTERS) \
		--clustering $(ALGORITHM) \
		--features $(FEATURES) \
		--labels $(LABELS)
	@echo "Figure generation complete. Figures saved to $(OUTPUT_DIR)/"
	@echo "Available figures:"
	@ls -la $(OUTPUT_DIR)/*.png
	@echo "Caption file: $(OUTPUT_DIR)/figure_captions.txt"

# Clean generated files
clean:
	@echo "Cleaning up generated figures..."
	@rm -rf $(OUTPUT_DIR)
	@echo "Done."