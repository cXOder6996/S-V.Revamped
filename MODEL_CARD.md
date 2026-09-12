# Model Card for SkinVision

## Model Details
- **Architecture**: EfficientNetB0 (ImageNet pre-trained weights)
- **Framework**: TensorFlow 2.20.0 / Keras 3.11.3
- **Input Processing**: Raw RGB images, float [0, 255] (Model handles internal Rescaling layer. EfficientNetB0 `preprocess_input` is a NO-OP in TF 2.20.0/Keras 3.11.3). 
- **Output**: 7 classes representing various skin lesions.
- **Model format**: `.keras` (Not `.h5`)

## Intended Use
The model is intended to classify dermoscopy images of skin lesions into one of 7 categories: 
- `akiec` (Actinic Keratoses)
- `bcc` (Basal Cell Carcinoma)
- `bkl` (Benign Keratosis)
- `df` (Dermatofibroma)
- `mel` (Melanoma)
- `nv` (Melanocytic Nevi)
- `vasc` (Vascular Lesions)

**Important**: The model is trained strictly on **dermoscopic images only**. The model outputs class probabilities/confidence based on visual similarity to the training classes. It does **not** calculate a clinical "risk level". All evaluations for calibration and confidence derive from the validation split. 

## Metrics
*(To be filled after training evaluation)*
- **Accuracy**: TBA
- **F1-Score (Macro and Weighted)**: TBA
- **Precision & Recall per class**: TBA

## Limitations
- **Data Distribution**: The classes `df` (Dermatofibroma) and `vasc` (Vascular Lesions) are statistically unstable due to very low sample counts in the HAM10000 dataset. Predictions and confidence estimates for these classes should be interpreted with caution.
- **Image Type**: Only dermoscopy images are supported. General clinical (macroscopic) images are out of distribution, and predictions on such images are invalid.
- **Missing Model**: No random prediction fallback is utilized. The system will raise errors if the model artifact is missing.
