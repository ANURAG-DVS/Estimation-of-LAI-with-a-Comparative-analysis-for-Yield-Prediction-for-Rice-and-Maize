document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const uploadedImageContainer = document.getElementById('uploadedImageContainer');
    const uploadedImage = document.getElementById('uploadedImage');
    const resultsContainer = document.getElementById('resultsContainer');
    const errorMessage = document.getElementById('errorMessage');
    const metricsTable = document.getElementById('metricsTable');
    const resultChartCtx = document.getElementById('resultChart').getContext('2d');

    let resultChart;

    // Simulate an API call to the ML model
    const analyzeImage = async (file) => {
        // Simulated response for demo purposes
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({
                    metrics: [
                        { metric: 'Leaf Area Index (LAI)', rice: '4.5', maize: '3.8' },
                        { metric: 'Predicted Yield (kg/ha)', rice: '6000', maize: '5400' }
                    ],
                    graphData: {
                        labels: ['January', 'February', 'March', 'April', 'May'],
                        rice: [1000, 2000, 3000, 4000, 5000],
                        maize: [900, 1800, 2700, 3600, 4500]
                    }
                });
            }, 2000);
        });
    };

    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('bg-gray-50');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('bg-gray-50');
    });

    dropzone.addEventListener('drop', async (e) => {
        e.preventDefault();
        dropzone.classList.remove('bg-gray-50');

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            displayUploadedImage(file);
            errorMessage.classList.add('hidden');
            const results = await analyzeImage(file);
            displayResults(results);
        } else {
            showError('Please upload a valid image file.');
        }
    });

    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file && file.type.startsWith('image/')) {
            displayUploadedImage(file);
            errorMessage.classList.add('hidden');
            const results = await analyzeImage(file);
            displayResults(results);
        } else {
            showError('Please upload a valid image file.');
        }
    });

    const displayUploadedImage = (file) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImage.src = e.target.result;
            uploadedImageContainer.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    };

    const displayResults = ({ metrics, graphData }) => {
        resultsContainer.classList.remove('hidden');

        // Populate Metrics Table
        metricsTable.innerHTML = metrics.map(row => `
            <tr>
                <td class="border border-gray-300 px-4 py-2">${row.metric}</td>
                <td class="border border-gray-300 px-4 py-2">${row.rice}</td>
                <td class="border border-gray-300 px-4 py-2">${row.maize}</td>
            </tr>
        `).join('');

        // Render Graph
        if (resultChart) resultChart.destroy();
        resultChart = new Chart(resultChartCtx, {
            type: 'line',
            data: {
                labels: graphData.labels,
                datasets: [
                    {
                        label: 'Rice Yield',
                        data: graphData.rice,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    },
                    {
                        label: 'Maize Yield',
                        data: graphData.maize,
                        borderColor: 'rgba(255, 159, 64, 1)',
                        backgroundColor: 'rgba(255, 159, 64, 0.2)',
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                },
            }
        });
    };

    const showError = (message) => {
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
    };
});