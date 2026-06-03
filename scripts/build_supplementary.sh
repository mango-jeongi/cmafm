#!/bin/bash
# scripts/build_supplementary.sh
# Packages the repository for BMVC 2026 double-blind submission.
# Ensures the zip is under the 100MB limit and completely anonymized.

cd "$(dirname "$0")/.." # Move to repo root
ZIP_PATH="$(pwd)/../Supplementary.zip"
echo "Packaging submission to $ZIP_PATH..."

# Remove old zip if exists
rm -f "$ZIP_PATH"

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
cp -r . $TEMP_DIR/code

# Scrub PII, large files, and caches
cd $TEMP_DIR
# Remove git
find . -name ".git" -exec rm -rf {} + 2>/dev/null
# Remove environment, local variables, and caches
find . -name ".venv" -exec rm -rf {} + 2>/dev/null
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name ".env" -exec rm -f {} + 2>/dev/null
# Remove weights to keep under 100MB (users must download them)
find . -name "*.pt" -exec rm -f {} + 2>/dev/null
# Remove the cloned engine (setup.sh will clone it for the reviewers)
rm -rf code/cft_engine
# Remove massive runs and logs folders to guarantee <100MB
rm -rf code/runs code/logs

# Zip the sanitized code
zip -rq Supplementary.zip code/
mv Supplementary.zip "$ZIP_PATH"

# Cleanup
cd - > /dev/null
rm -rf $TEMP_DIR

echo "Done! Final size of Supplementary.zip:"
du -sh "$ZIP_PATH"
