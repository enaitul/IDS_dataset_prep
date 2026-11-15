.PHONY: check test all
check: ; python check_raw.py
test:  ; python -m pytest tests -q
all:   ; python run_pipeline.py
