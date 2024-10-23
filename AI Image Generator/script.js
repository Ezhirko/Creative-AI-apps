document.addEventListener('DOMContentLoaded', () => {
    const generatedImage = document.getElementById('generatedImage');
    const generatedImageBox = document.getElementById('generatedImageBox');
    const generateBtn = document.getElementById('generateBtn');
    const promptInput = document.getElementById('promptInput');
    const fileInput = document.getElementById('fileInput');
    const selectedImage = document.getElementById('selectedImage');
    const imageViewer = document.getElementById('imageViewer');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const fileType = document.getElementById('fileType');

    generateBtn.addEventListener('click', generateImage);
    fileInput.addEventListener('change', displaySelectedImage);

    // Function to check if the image is empty
    function updateImageState(imageElement, containerElement) {
        if (imageElement.src && imageElement.src !== window.location.href) {
            containerElement.classList.remove('empty');
        } else {
            containerElement.classList.add('empty');
        }
    }

    // Call this function initially to set the correct state for both images
    updateImageState(generatedImage, generatedImageBox);
    updateImageState(selectedImage, imageViewer);

    // Add event listeners for image load and error
    generatedImage.addEventListener('load', () => updateImageState(generatedImage, generatedImageBox));
    generatedImage.addEventListener('error', () => updateImageState(generatedImage, generatedImageBox));
    selectedImage.addEventListener('load', () => updateImageState(selectedImage, imageViewer));
    selectedImage.addEventListener('error', () => updateImageState(selectedImage, imageViewer));

    function generateImage() {
        const selectedAnimal = document.querySelector('input[name="animal"]:checked').value;
        const prompt = promptInput.value;

        // Make API call to Flask backend
        fetch('http://localhost:5000/generate_image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
            body: JSON.stringify({ animal: selectedAnimal, prompt: prompt }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.image_url) {
                generatedImage.src = data.image_url;
                generatedImage.alt = `Generated ${selectedAnimal} image`;
            } else if (data.message) {
                alert(data.message);
            } else {
                alert('Failed to generate image');
                generatedImage.src = '';
                generatedImage.alt = '';
            }
            updateImageState(generatedImage, generatedImageBox);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while generating the image');
            generatedImage.src = '';
            generatedImage.alt = '';
            updateImageState(generatedImage, generatedImageBox);
        });
    }

    function displaySelectedImage(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                selectedImage.src = e.target.result;
                selectedImage.alt = file.name;
            };
            reader.readAsDataURL(file);

            // Display file details
            fileName.textContent = file.name;
            fileSize.textContent = formatFileSize(file.size);
            fileType.textContent = file.type;
        } else {
            // If no file is selected, clear the image and update the state
            selectedImage.src = '';
            selectedImage.alt = '';
            updateImageState(selectedImage, imageViewer);
        }
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' bytes';
        else if (bytes < 1048576) return (bytes / 1024).toFixed(2) + ' KB';
        else return (bytes / 1048576).toFixed(2) + ' MB';
    }
});
