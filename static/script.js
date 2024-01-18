function processImage() {
    const input = document.getElementById('imageInput');
    const statusMessage = document.getElementById('statusMessage');

    // Check if a file is selected
    if (input.files.length > 0) {
        const file = input.files[0];

        const reader = new FileReader();

        reader.onload = function (e) {
            const formData = new FormData();
            formData.append('image', file);

            fetch('/process', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(result => {
                // Update the status message with the classification text
                statusMessage.innerText = result.status_message;
                statusMessage.style.color = result.classification === 'It\'s a Tumor' ? '#27ae60' : '#e74c3c'; // Green for Tumor, Red for No Tumor

                // Open a new tab and display the processed image
                const newTab = window.open('', '_blank');
                newTab.document.write('<html><head><title>Processed Image</title></head><body>');
                newTab.document.write(`<img src="${result.result_image_path}" alt="Processed Image">`);
                newTab.document.write('</body></html>');
            })
            .catch(error => console.error('Error:', error));
        };

        // Read the selected image as a data URL
        reader.readAsDataURL(file);
    } else {
        // Update the status message for no image selected
        statusMessage.innerText = 'Please select an image.';
        statusMessage.style.color = '#e74c3c'; // Red color for error
    }
}
