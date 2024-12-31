document.addEventListener('DOMContentLoaded', function () {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsContainer = document.getElementById('resultsContainer');
    const predictionResult = document.getElementById('predictionResult');

    // Handle click to upload
    dropZone.addEventListener('click', () => fileInput.click());

    // Handle file selection
    fileInput.addEventListener('change', handleFileSelect);

    // Handle drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload a valid image file.');
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('hidden');
            uploadImage(file);
        };
        reader.readAsDataURL(file);
    }

    async function uploadImage(file) {
        // Show loading state
        loadingSpinner.classList.remove('hidden');
        resultsContainer.classList.add('hidden');

        const formData = new FormData();
        formData.append('image', file);

        try {
            // Send image to Flask backend
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();

            // Hide loading state
            loadingSpinner.classList.add('hidden');

            // Show results
            if (data.error) {
                predictionResult.textContent = 'Error: ' + data.error;
            } else {
                predictionResult.innerHTML = `<strong>Crop Prediction:</strong> ${data.prediction}<br><strong>LAI Prediction:</strong> ${data.lai_prediction.toFixed(4)}`;
            }
            resultsContainer.classList.remove('hidden');
        } catch (error) {
            console.error('Error:', error);
            loadingSpinner.classList.add('hidden');
            predictionResult.textContent = 'Error processing image. Please try again.';
            resultsContainer.classList.remove('hidden');
        }
    }
});
