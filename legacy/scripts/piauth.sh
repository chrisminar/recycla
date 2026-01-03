#!/usr/bin/env bash

# ========================================
# Pi Authentication Setup Script
# ========================================
# This script sets up authentication credentials and configuration for a Raspberry Pi
# device in the Recycla system. It downloads service account keys from Google Cloud
# Storage, generates a Pi ID, and configures environment variables.

# Activate the Python virtual environment (version 11)
# This ensures we're using the correct Python packages and dependencies
source .venv11/bin/activate

# Change to the scripts directory to ensure relative paths work correctly
cd scripts

# Remind user to authenticate with Google Cloud CLI first
# This is required before we can download files from Google Cloud Storage
echo "did you remember to gcloud auth login?"

# ========================================
# Download Service Account Key
# ========================================
# Set up the destination path for storing secrets locally
dst_path="../.secrets"
echo "Making secrets directory at $dst_path"
# Create the secrets directory if it doesn't exist (mkdir -p won't fail if dir exists)
mkdir -p $dst_path # make secrets directory if it doesn't exist

# Define the source path in Google Cloud Storage where the service account key is stored
# the service account key is a service account with access to the recycla project so the pi can upload to firebase
src_path="gs://recyclo-secrets/service_account_key.json"

# Check if we already have the service account key locally
if [ -f "$dst_path/service_account_key.json" ]; then
    echo "service_account_key.json already exists in $dst_path"
else
    # Download the service account key from Google Cloud Storage if we don't have it
    echo "service_account_key.json does not exist in $dst_path"
    echo "Trying to grab json from $src_path"
    gcloud storage cp $src_path $dst_path
fi

# ========================================
# Generate Pi ID and Configuration
# ========================================
# Run the Pi helpers script to generate authentication configuration
# This creates a piauth.json file with the Pi's unique identifier
echo "Using python to get pi id"
python3 ../recycla/src/recycla/connect/pi_helpers.py

# Extract the Pi ID from the generated piauth.json file
# This uses inline Python to parse the JSON and extract the pi_id field
pi_id=$(python3 -c "import json; f = open('$dst_path/piauth.json'); data = json.load(f); f.close(); print(data['pi_id'])")

# Export the PI_ID as an environment variable for the current session
export PI_ID=$pi_id

# ========================================
# Persistent Environment Variable Setup
# ========================================
# Add the PI_ID export to .bashrc so it persists across sessions
# We need to be careful to maintain the existing structure of .bashrc
# The raspberry pi is set up to call wrapper.sh as the last line of .bashrc
# this makes the script get launched on terminal open & boot

# Get the last line of .bashrc (which should be the wrapper.sh call)
last_line=$(tail -n 1 ~/.bashrc)

# Check if the last line contains wrapper.sh AND we haven't already added PI_ID export
if [[ "$last_line" == *"wrapper.sh"* ]] && ! grep -q "export PI_ID=" ~/.bashrc; then
    # Remove the last line (wrapper.sh call) temporarily
    sed -i '$ d' ~/.bashrc
    
    # Add the PI_ID export line
    echo "export PI_ID=$PI_ID" >> ~/.bashrc
    
    # Re-add the wrapper.sh call as the last line
    echo "$last_line" >> ~/.bashrc
    
    echo "Modifying .bashrc"
    # Reload .bashrc to apply changes immediately
    source ~/.bashrc
fi

# Confirm the PI_ID has been set
echo "PI_ID set to $PI_ID"
