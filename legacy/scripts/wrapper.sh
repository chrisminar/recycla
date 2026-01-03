#!/bin/bash

# ========================================
# Recycla Service Wrapper Script
# ========================================
# This script is designed to run the Recycla service on a Raspberry Pi or other device.
# It handles automatic updates, dependency management, authentication setup, and keeps
# the service running continuously with automatic restarts on failure.

# Parse command line argument for development mode (defaults to false if not provided)
# Usage: ./wrapper.sh true (for develop mode) or ./wrapper.sh (for production mode)
develop_mode=${1:-false}

# ========================================
# Initial Setup and Cleanup
# ========================================
# Navigate to the main project directory
echo "Change dir"
cd ~/Documents/src/recycla

# Kill any existing Recycla processes to prevent conflicts
# This ensures we start with a clean slate
echo "kill previous client.py process"
pkill -f "python.*recycla"

# Activate the Python virtual environment (version 11)
# This isolates our Python dependencies from the system Python
echo "Activate venv"
source .venv11/bin/activate
    
# ========================================
# Time-Based Installation Strategy
# ========================================
# The script uses different installation strategies based on the time of day
# Early morning hours (12-3 AM PDT) are used for more aggressive updates
# This is because the update takes a long time on the pis and it was not ideal
# to have the pi go down for 30 minutes during a mid day power cycle
echo "Checking current time to determine install strategy..."

# Detect the system timezone for logging purposes
system_timezone=$(timedatectl show --property=Timezone --value 2>/dev/null || echo "Unknown")
echo "System timezone: $system_timezone"

# ========================================
# Cross-Platform Time Detection
# ========================================
# Convert current time to Pacific Daylight Time (PDT) for consistent scheduling
# This ensures the update schedule works regardless of where the Pi is deployed
echo "Using system time converted to PDT"

# Try different methods to get PDT time based on the operating system
if command -v timedatectl >/dev/null 2>&1; then
    # Linux system with systemd (most modern Linux distributions)
    current_hour=$(TZ='America/Los_Angeles' date +"%H")
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS system
    current_hour=$(TZ='America/Los_Angeles' date +"%H")
else
    # Fallback method: manually calculate PDT/PST from UTC
    # PDT is UTC-7 (March to November), PST is UTC-8 (December to February)
    
    # Get current month to determine if we're in PDT or PST
    month=$(date +"%m")
    if [ "$month" -ge 3 ] && [ "$month" -le 11 ]; then
        # Pacific Daylight Time (UTC-7)
        utc_hour=$(date -u +"%H")
        current_hour=$(( (utc_hour - 7 + 24) % 24 ))
    else
        # Pacific Standard Time (UTC-8)
        utc_hour=$(date -u +"%H")
        current_hour=$(( (utc_hour - 8 + 24) % 24 ))
    fi
fi

echo "Current PDT hour: $current_hour"

# ========================================
# Installation Strategy Based on Time
# ========================================
# Use different pip install flags based on the time of day to balance
# performance and reliability. Early morning hours allow for more thorough updates.

# Check if we're in the early morning maintenance window (12 AM to 3 AM PDT)
# During this time, we use more aggressive update strategies
if [ "$current_hour" -ge 0 ] && [ "$current_hour" -le 3 ]; then
    echo "Time is between 12 AM and 3 AM PDT - using --force-reinstall"
    # Force reinstall all packages and their dependencies (slower but more thorough)
    force_flag="--force-reinstall"
else
    echo "Time is between 3 AM and 12 AM PDT - using standard install"
    # Force reinstall but skip dependencies (faster for normal operation)
    force_flag="--force-reinstall --no-deps"
fi

# ========================================
# Git Repository Updates
# ========================================
# During the maintenance window, also pull the latest code from git
# This ensures we're running the most up-to-date version of the software

if [ "$current_hour" -ge 0 ] && [ "$current_hour" -le 3 ]; then
    echo "Time is between 12 AM and 3 AM PDT repulling from git"
    
    if [ "$develop_mode" = true ]; then
        # In development mode, use the develop branch for latest features
        echo "Pulling develop branch"
        git checkout develop
        git pull
    else
        # In production mode, use the main branch for stability
        echo "Pulling main branch"
        git checkout main -f  # -f flag forces checkout even if there are local changes
        git pull
    fi
fi

# ========================================
# Git Authentication Detection
# ========================================
# Detect how git is configured to authenticate with the repository
# This allows pip to install from git using the same authentication method

echo "Detecting git authentication method..."

# Get the remote URL to determine what authentication method is being used
remote_url=$(git remote get-url origin 2>/dev/null || echo "")

# Construct the appropriate git URL for pip based on the authentication method
if [[ "$remote_url" == *"ssh://"* ]] || [[ "$remote_url" == *"git@"* ]]; then
    echo "SSH authentication detected"
    # Use SSH format for pip installation (requires SSH keys to be set up)
    git_url="git+ssh://git@github.com/chrisminar/recycla.git"
elif [[ "$remote_url" == *"https://"* ]]; then
    echo "HTTPS authentication detected"
    # Use the existing HTTPS URL (may use stored credentials or tokens)
    git_url="git+${remote_url}"
elif [[ "$remote_url" == *"http://"* ]]; then
    echo "HTTP authentication detected"
    # Use the existing HTTP URL (less secure, rarely used)
    git_url="git+${remote_url}"
else
    echo "Unknown authentication method, defaulting to HTTPS"
    # Fallback to public HTTPS URL
    git_url="git+https://github.com/chrisminar/recycla.git"
fi
echo "Using git URL: $git_url"

# ========================================
# Recycla Package Installation
# ========================================
# Install the Recycla package from the git repository
# The installation method depends on whether we're in development or production mode

if [ "$develop_mode" = true ]; then
    echo "Installing from develop branch"
    # In development mode, install directly from the develop branch
    # This gives us the latest unreleased features and bug fixes
    pip install --no-input $force_flag $git_url@develop#subdirectory=recycla
else
    echo "Fetch latest tags to find the latest release"
    # In production mode, install from the latest tagged release for stability
    git fetch --tags

    echo "Get the latest release tag"
    # Find the most recent git tag (which should be the latest release)
    latest_tag=$(git describe --tags `git rev-list --tags --max-count=1`)

    echo "Installing latest release tag: $latest_tag"
    # Install the specific tagged version from the recycla subdirectory
    pip install --no-input $force_flag $git_url@$latest_tag#subdirectory=recycla
fi

# ========================================
# Environment Variable Configuration
# ========================================
# Set up required environment variables for the Recycla service

# Set the root path so the application knows where to find configuration files
export RECYCLA_ROOT_PATH=$(pwd)
echo "RECYCLA_ROOT_PATH set to $RECYCLA_ROOT_PATH"

# Configure Google Cloud authentication credentials
# This allows the service to authenticate with Google Cloud services
export GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/.secrets/service_account_key.json
echo "GOOGLE_APPLICATION_CREDENTIALS set to $GOOGLE_APPLICATION_CREDENTIALS"

# Activate the Google Cloud service account for this session
echo "Authenticate gcloud with the service account key"
gcloud auth activate-service-account --key-file .secrets/service_account_key.json

# ========================================
# Model Updates (Maintenance Window Only)
# ========================================
# Check for and download updated machine learning models during the maintenance window
# This is done during off-peak hours because it can take a long time and use bandwidth

if [ "$current_hour" -ge 0 ] && [ "$current_hour" -le 3 ]; then
    # Model updates are time-consuming and not needed during development/testing
    echo "checking for new models"
    recycla update-model
fi

# ========================================
# Main Service Loop
# ========================================
# Start the main Recycla service and keep it running continuously
# If the service crashes or exits, automatically restart it after a brief delay

while true; do
    echo "Run script"
    # Run the main Recycla service, capturing both stdout and stderr
    # The 2>&1 redirects stderr to stdout so we see all output
    recycla run 2>&1

    # If we reach this point, the service has exited (either normally or due to an error)
    echo "client.py exited. Restarting in 5 seconds..."
    
    # Wait 5 seconds before restarting to prevent rapid restart loops
    # This gives time for any temporary issues to resolve
    sleep 5
done

# This line should never be reached due to the infinite loop above
# It's here as a safety net in case the loop is somehow broken
read -p "Press Enter to close"
