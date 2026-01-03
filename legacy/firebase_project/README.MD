# Firebase Functions - Recycla Project

This directory contains Firebase Cloud Functions for the Recycla waste classification system. These functions handle real-time processing of video data, user notifications, device monitoring, and analytics.

## Overview

The Firebase Functions provide backend processing for a smart waste classification system that uses IoT devices (Raspberry Pis) to capture and classify recyclable materials. The system includes machine learning inference, OCR text processing for grocery items, push notifications, and comprehensive analytics.

## Functions Architecture

### Core Functions (main.py)

#### `on_video_create`
- **Trigger**: New document created in `videos/{id}` collection
- **Memory**: 4GB (for ML model inference)
- **Purpose**: Processes newly uploaded videos through machine learning classification
- **Process**:
  1. Downloads video from Google Cloud Storage
  2. Extracts frames for analysis
  3. Runs ML inference using PyTorch model
  4. Updates document with classification results and confidence scores

#### `on_video_update`
- **Trigger**: Document updated in `videos/{id}` collection  
- **Memory**: 512MB
- **Purpose**: Handles user ground truth updates and grocery item processing
- **Process**:
  1. Detects changes to `user_ground_truth` or `class_id` fields
  2. Updates PI summary statistics when classifications change
  3. Processes OCR text extraction for grocery items when user provides ground truth
  4. Skips processing for non-interesting materials

#### `count_documents_hourly`
- **Trigger**: Scheduled every 60 minutes
- **Memory**: 512MB
- **Purpose**: Maintains document counts for system monitoring
- **Process**:
  1. Counts documents in `videos` and `pis` collections
  2. Stores counts in `stats/hourly_counts` document with timestamp

#### `check_pi_health_and_update_status`
- **Trigger**: Scheduled every 30 minutes
- **Memory**: 512MB
- **Purpose**: Monitors device connectivity and health
- **Process**:
  1. Checks health timestamps in Realtime Database
  2. Sets devices offline if no health update in 10+ minutes
  3. Triggers disconnect notifications when status changes

#### `on_device_connection_status_change`
- **Trigger**: Realtime Database value change at `/devices/{pi_id}/connection_status`
- **Memory**: 512MB
- **Purpose**: Sends push notifications when devices disconnect
- **Process**:
  1. Detects status changes to "offline"
  2. Looks up users associated with the device
  3. Sends FCM push notifications to all device users

#### `on_broadcast_created_send_push`
- **Trigger**: New document created in `broadcasts/{id}` collection
- **Memory**: 512MB
- **Purpose**: Sends broadcast messages to all users
- **Process**:
  1. Validates broadcast message content
  2. Retrieves all users with FCM tokens
  3. Sends push notifications to all eligible users
  4. Handles token cleanup for invalid/expired tokens

## Supporting Modules

### inference.py
**Machine Learning Classification Engine**

- **Model Loading**: Downloads PyTorch models from Google Cloud Storage with retry logic
- **Video Processing**: Extracts frames from uploaded videos using OpenCV
- **Classification**: Runs inference using trained recycling classification models
- **Error Handling**: Comprehensive retry mechanisms for network failures
- **Memory Management**: Efficient handling of large ML models in serverless environment

**Key Functions**:
- `run_inference()`: Main classification pipeline
- `download_video_from_gcs()`: Secure video download with retry
- `load_model_from_gcs()`: Model loading with caching
- `initialize_model_and_classnames()`: Cold start optimization

### reading.py  
**OCR and Text Processing**

- **Google Vision API**: Extracts text from recycling item images
- **Gemini AI Integration**: Converts OCR text to grocery list items
- **Confidence Scoring**: Provides reliability scores for text recognition
- **Grocery List Generation**: Creates formatted shopping lists from detected items

**Key Functions**:
- `detect_text()`: OCR text extraction from images
- `gemini_item_from_text()`: AI-powered grocery item identification
- `image_to_item()`: Complete pipeline from image to grocery item
- `gemini_grocery_list()`: Formats multiple items into shopping lists

### broadcast.py
**Push Notification System**

- **FCM Integration**: Firebase Cloud Messaging for mobile notifications
- **Token Management**: Automatic cleanup of invalid/expired tokens
- **Error Handling**: Comprehensive FCM error handling and retry logic
- **User Targeting**: Broadcasts to all users with valid FCM tokens

**Features**:
- Validates message content (title, body required)
- Handles FCM-specific errors (unregistered tokens, quota limits)
- Automatic token cleanup for better delivery rates
- Detailed success/failure reporting

### device_disconnect.py
**IoT Device Monitoring**

- **Health Checking**: Monitors device heartbeat timestamps
- **Status Management**: Updates connection status in Realtime Database  
- **User Notifications**: Alerts users when their devices go offline
- **Automated Recovery**: Handles device reconnection scenarios

**Key Functions**:
- `check_health()`: Scheduled health monitoring
- `device_disconnect()`: Handles disconnect events and notifications

### pi_summary.py
**Analytics and Reporting**

- **Statistics Calculation**: Computes classification accuracy metrics
- **Time-based Analysis**: Weekly and monthly performance breakdowns
- **Material Classification**: Tracks performance by material type
- **Bounty Validation**: Calculates valid items for reward systems

**Analytics Computed**:
- Overall classification accuracy percentages
- Material-specific performance metrics
- Time-series analysis (weekly/monthly trends)
- Valid item counts for bounty calculations

### fb_help.py
**Database Utilities and Helpers**

- **Firebase Initialization**: Handles authentication and client setup
- **Data Validation**: Label mapping and classification verification
- **Material Processing**: Material and sub-material extraction logic
- **Utility Functions**: Date formatting, document counting, video retrieval

**Key Features**:
- Automatic Firebase authentication (service account or default credentials)
- Material label mapping for consistent classification
- Prediction accuracy validation logic
- Helper functions for date/time processing

### count_documents.py
**System Monitoring**

- **Document Counting**: Efficient aggregation queries for collection sizes
- **Statistics Storage**: Persistent tracking of system growth
- **Monitoring Integration**: Data for system health dashboards

## Dependencies

The functions require the following key dependencies:

```python
firebase_functions>=0.1.8    # Firebase Functions framework
firebase-admin               # Firebase Admin SDK
torch==2.7.1                # PyTorch for ML inference
torchvision                 # Computer vision utilities
opencv-python-headless      # Video processing
google-cloud-storage        # GCS integration
google-cloud-vision         # OCR capabilities
google-genai               # Gemini AI integration
numpy==1.26.4              # Numerical computing
```

## Configuration

### Firebase Project Setup
- Project ID: `recyclo-c0fd1`
- Realtime Database: `https://recyclo-c0fd1-default-rtdb.firebaseio.com/`
- Storage Bucket: `recyclo-c0fd1.firebasestorage.app`

### Authentication
Functions use Firebase service account authentication with fallback to Application Default Credentials for Cloud environments.

### Memory Allocation
- **Inference Functions**: 4GB (for ML model loading)
- **Standard Functions**: 512MB (for general processing)

## Data Flow

1. **Video Upload**: IoT devices upload videos to Cloud Storage
2. **Document Creation**: Firestore document created with video metadata
3. **ML Processing**: `on_video_create` triggers classification inference
4. **User Interaction**: Users can provide ground truth corrections
5. **Analytics Update**: `on_video_update` recalculates statistics
6. **OCR Processing**: Text extraction for grocery items (if applicable)
7. **Monitoring**: Scheduled functions track system health and usage

## Error Handling

- **Retry Logic**: Exponential backoff for transient failures
- **Graceful Degradation**: Functions continue operation with partial failures
- **Comprehensive Logging**: Detailed error reporting for debugging
- **Token Cleanup**: Automatic removal of invalid FCM tokens
- **Resource Management**: Proper cleanup of temporary files and connections

## Security

- **Service Account Authentication**: Secure credential management
- **Input Validation**: Comprehensive validation of all user inputs
- **Error Sanitization**: Safe error messages without sensitive data exposure
- **Token Management**: Secure FCM token handling and cleanup

## Deployment

Deploy functions using Firebase CLI:

```bash
firebase deploy --only functions
```

Functions are automatically deployed to Google Cloud with the specified memory and trigger configurations.

## Monitoring and Maintenance

- **Cloud Functions Logs**: Monitor function execution and errors
- **Firestore Rules**: Ensure proper database security
- **Performance Monitoring**: Track function execution times and memory usage
- **Cost Optimization**: Monitor function invocations and resource usage