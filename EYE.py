 
    <!-- Contact Modal -->
    <div id="contactModal" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 hidden">
        <div class="bg-white rounded-lg p-6 max-w-md w-full">
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-xl font-bold text-gray-800">Contact Information</h3>
                <button id="closeContact" class="text-gray-500 hover:text-gray-700">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="space-y-4">
                <div>
                    <p class="font-medium text-gray-700">Name:</p>
                    <p class="text-gray-600">Hrishi Kamalachandran</p>
                </div>
                <div>
                    <p class="font-medium text-gray-700">Email:</p>
                    <p class="text-gray-600">hrishi.kamal16@gmail.com</p>
                </div>
                <div>
                    <p class="font-medium text-gray-700">Description:</p>
                    <p class="text-gray-600">A Current High School Senior Interested In Pursuing The Medical Field</p>
                </div>
                <div class="pt-4 border-t border-gray-200">
                    <p class="font-medium text-gray-700 mb-2">Connect:</p>
                    <div class="flex space-x-4">
                        <a href="https://www.instagram.com/hrishi_kamal/" target="_blank" class="text-pink-600 hover:text-pink-700">
                            <i class="fab fa-instagram text-2xl"></i>
                        </a>
                        <a href="https://www.linkedin.com/in/hrishi-kamalachandran-196669288/" target="_blank" class="text-blue-600 hover:text-blue-700">
                            <i class="fab fa-linkedin text-2xl"></i>
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Contact modal functionality
        const contactLink = document.getElementById('contactLink');
        const contactModal = document.getElementById('contactModal');
        const closeContact = document.getElementById('closeContact');

        contactLink.addEventListener('click', (e) => {
            e.preventDefault();
            contactModal.classList.remove('hidden');
        });

        closeContact.addEventListener('click', () => {
            contactModal.classList.add('hidden');
        });

        // Close modal when clicking outside
        contactModal.addEventListener('click', (e) => {
            if (e.target === contactModal) {
                contactModal.classList.add('hidden');
            }
        });
    </script>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clinical Eye Disorder Classifier</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        :root {
            --primary: #4f46e5;
            --primary-dark: #4338ca;
            --secondary: #10b981;
            --dark: #1e293b;
            --darker: #0f172a;
            --light: #f8fafc;
            --danger: #ef4444;
        }
        
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f1f5f9;
            color: var(--dark);
        }
        
        .gradient-bg {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        
        .disease-card {
            transition: all 0.3s ease;
        }
        
        .disease-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        }
        
        .probability-bar {
            height: 8px;
            border-radius: 4px;
            background-color: #e2e8f0;
        }
        
        .probability-fill {
            height: 100%;
            border-radius: 4px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            transition: width 0.5s ease;
        }
        
        .upload-area {
            border: 2px dashed #cbd5e1;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: var(--primary);
            background-color: rgba(79, 70, 229, 0.05);
        }
        
        .upload-area.dragover {
            border-color: var(--primary);
            background-color: rgba(79, 70, 229, 0.1);
        }
        
        .diagnosis-badge {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% {
                box-shadow: 0 0 0 0 rgba(79, 70, 229, 0.4);
            }
            70% {
                box-shadow: 0 0 0 10px rgba(79, 70, 229, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(79, 70, 229, 0);
            }
        }
        
        .sidebar {
            background: linear-gradient(180deg, var(--darker) 0%, var(--dark) 100%);
        }
        
        .nav-link {
            transition: all 0.2s ease;
        }
        
        .nav-link:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        
        .nav-link.active {
            background-color: rgba(255, 255, 255, 0.15);
        }
    </style>
</head>
<body class="min-h-screen">
    <!-- Navigation -->
    <nav class="gradient-bg text-white shadow-lg">
        <div class="container mx-auto px-4 py-3 flex justify-between items-center">
            <div class="flex items-center space-x-2">
                <i class="fas fa-eye text-2xl"></i>
                <span class="text-xl font-bold">Clinical Eye Disorder Classifier</span>
            </div>
            <div class="hidden md:flex items-center space-x-6">
                <a href="#" class="hover:underline">Home</a>
                <a href="#" id="contactLink" class="hover:underline">Contact</a>
            </div>
            <button class="md:hidden text-white focus:outline-none">
                <i class="fas fa-bars text-2xl"></i>
            </button>
        </div>
    </nav>

    <div class="flex flex-col lg:flex-row">
        <!-- Sidebar -->
        <div class="sidebar text-white w-full lg:w-64 p-4 lg:min-h-screen">
            <div class="mb-8">
                <h3 class="text-lg font-semibold mb-2">System Parameters</h3>
                <p class="text-sm text-gray-300">Powered by deep learning CNN model</p>
            </div>
            
            <div class="mb-6">
                <h4 class="font-medium mb-3 flex items-center">
                    <i class="fas fa-camera mr-2"></i> Image Guidelines
                </h4>
                <ul class="space-y-2 text-sm">
                    <li class="flex items-start">
                        <i class="fas fa-check-circle text-green-400 mt-1 mr-2"></i>
                        <span>Position camera at arm's length</span>
                    </li>
                    <li class="flex items-start">
                        <i class="fas fa-check-circle text-green-400 mt-1 mr-2"></i>
                        <span>Use even, diffused lighting</span>
                    </li>
                    <li class="flex items-start">
                        <i class="fas fa-check-circle text-green-400 mt-1 mr-2"></i>
                        <span>Crop tightly around the eye</span>
                    </li>
                    <li class="flex items-start">
                        <i class="fas fa-check-circle text-green-400 mt-1 mr-2"></i>
                        <span>Maintain sharp focus on iris</span>
                    </li>
                </ul>
            </div>
            
            <div class="mb-6">
                <h4 class="font-medium mb-3 flex items-center">
                    <i class="fas fa-vial mr-2"></i> Possible Eye Conditions
                </h4>
                <div class="space-y-3">
                    <div class="bg-slate-700 bg-opacity-50 rounded-lg p-3">
                        <p class="font-medium">Cataract</p>
                        <p class="text-xs text-gray-300">Clouding of the natural lens</p>
                    </div>
                    <div class="bg-slate-700 bg-opacity-50 rounded-lg p-3">
                        <p class="font-medium">Conjunctivitis</p>
                        <p class="text-xs text-gray-300">Inflammation of conjunctiva</p>
                    </div>
                    <div class="bg-slate-700 bg-opacity-50 rounded-lg p-3">
                        <p class="font-medium">Swelling</p>
                        <p class="text-xs text-gray-300">Eyelid or periorbital swelling</p>
                    </div>
                    <div class="bg-slate-700 bg-opacity-50 rounded-lg p-3">
                        <p class="font-medium">Uveitis</p>
                        <p class="text-xs text-gray-300">Inflammation of the uvea</p>
                    </div>
                </div>
            </div>
            
        </div>

        <!-- Main Content -->
        <div class="flex-1 p-6">
            <div class="max-w-6xl mx-auto">
                <div class="text-center mb-8">
                    <h1 class="text-3xl md:text-4xl font-bold text-gray-800 mb-2">Clinical Eye Disorder Classifier</h1>
                    <p class="text-gray-600">Upload an eye image for AI-powered disease prediction</p>
                </div>
                
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <!-- Upload Section -->
                    <div class="glass-card rounded-xl shadow-lg p-6">
                        <h2 class="text-xl font-semibold mb-4 text-gray-800">Image Acquisition</h2>
                        
                        <div id="uploadArea" class="upload-area rounded-lg p-8 text-center cursor-pointer mb-4">
                            <input type="file" id="fileInput" class="hidden" accept="image/*">
                            <div id="uploadContent">
                                <i class="fas fa-cloud-upload-alt text-4xl text-gray-400 mb-3"></i>
                                <p class="text-gray-600 mb-1">Drag & drop eye image here</p>
                                <p class="text-sm text-gray-500">or click to browse files</p>
                                <p class="text-xs text-gray-400 mt-3">Supports JPG, PNG, BMP (Max 5MB)</p>
                            </div>
                            <div id="imagePreview" class="hidden">
                                <img id="previewImage" src="" alt="Preview" class="max-h-64 mx-auto rounded-lg">
                                <button id="changeImage" class="mt-3 text-sm text-blue-600 hover:underline">Change image</button>
                            </div>
                        </div>
                        
                        <div class="mt-6">
                            <h3 class="font-medium mb-2">Image Guidelines</h3>
                            <div class="bg-blue-50 border border-blue-100 rounded-lg p-4">
                                <div class="flex items-start mb-2">
                                    <i class="fas fa-info-circle text-blue-500 mt-1 mr-2"></i>
                                    <span class="text-sm text-gray-700">For best results, ensure the eye is clearly visible with minimal reflections or shadows.</span>
                                </div>
                                <div class="flex items-start">
                                    <i class="fas fa-info-circle text-blue-500 mt-1 mr-2"></i>
                                    <span class="text-sm text-gray-700">The image should focus on one eye at a time, centered in the frame.</span>
                                </div>
                            </div>
                        </div>
                        
                        <button id="analyzeBtn" class="w-full mt-6 bg-gradient-to-r from-blue-600 to-blue-500 text-white py-3 rounded-lg font-medium shadow-md hover:shadow-lg transition duration-300 hidden">
                            Analyze Image
                        </button>
                    </div>
                    
                    <!-- Results Section -->
                    <div class="glass-card rounded-xl shadow-lg p-6">
                        <h2 class="text-xl font-semibold mb-4 text-gray-800">Diagnostic Analysis</h2>
                        
                        <div id="emptyState" class="text-center py-12">
                            <i class="fas fa-microscope text-4xl text-gray-300 mb-4"></i>
                            <p class="text-gray-500">Upload an eye image to begin analysis</p>
                        </div>
                        
                        <div id="resultsSection" class="hidden">
                            <div class="diagnosis-badge bg-gradient-to-r from-blue-100 to-blue-50 border border-blue-200 rounded-xl p-4 mb-6 text-center">
                                <p class="text-sm text-blue-600 mb-1">PRIMARY DIAGNOSIS</p>
                                <h3 id="diagnosisResult" class="text-2xl font-bold text-gray-800">Cataract</h3>
                                <p class="text-blue-600 font-medium">
                                    Confidence: <span id="confidenceValue" class="text-gray-800">87.5%</span>
                                </p>
                            </div>
                            
                            <div class="mb-6">
                                <h3 class="font-medium mb-3">Probability Distribution</h3>
                                <div id="probabilityBars" class="space-y-4">
                                    <!-- Probability bars will be inserted here by JavaScript -->
                                </div>
                            </div>
                            
                            <div>
                                <h3 class="font-medium mb-3">Clinical Notes</h3>
                                <div id="clinicalNotes" class="bg-gray-50 border border-gray-100 rounded-lg p-4">
                                    <p id="diseaseDescription" class="text-sm text-gray-700">
                                        Clouding of the natural lens of the eye, causing vision impairment. Cataracts typically develop slowly and can affect one or both eyes.
                                    </p>
                                    <div class="mt-3 pt-3 border-t border-gray-100">
                                        <p class="text-xs text-gray-500">
                                            <i class="fas fa-exclamation-triangle text-yellow-500 mr-1"></i>
                                            This is an AI prediction and should be verified by a medical professional.
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Model Information -->
                <div class="mt-12 glass-card rounded-xl shadow-lg p-6">
                    <h2 class="text-xl font-semibold mb-4 text-gray-800">Technical Specifications</h2>
                    
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div class="bg-white rounded-lg p-4 shadow-sm">
                            <div class="flex items-center mb-3">
                                <div class="bg-blue-100 p-2 rounded-lg mr-3">
                                    <i class="fas fa-project-diagram text-blue-600"></i>
                                </div>
                                <h3 class="font-medium">Model Architecture</h3>
                            </div>
                            <ul class="text-sm text-gray-600 space-y-2">
                                <li class="flex items-start">
                                    <i class="fas fa-layer-group text-gray-400 mt-1 mr-2 text-xs"></i>
                                    <span>4-layer convolutional neural network</span>
                                </li>
                                <li class="flex items-start">
                                    <i class="fas fa-expand text-gray-400 mt-1 mr-2 text-xs"></i>
                                    <span>Input: 224×224 RGB images</span>
                                </li>
                                <li class="flex items-start">
                                    <i class="fas fa-sign-out-alt text-gray-400 mt-1 mr-2 text-xs"></i>
                                    <span>Output: 5-class probability</span>
                                </li>
                            </ul>
                        </div>
                        
                        <div class="bg-white rounded-lg p-4 shadow-sm">
                            <div class="flex items-center mb-3">
                                <div class="bg-green-100 p-2 rounded-lg mr-3">
                                    <i class="fas fa-chart-line text-green-600"></i>
                                </div>
                                <h3 class="font-medium">Performance</h3>
                            </div>
                            <ul class="text-sm text-gray-600 space-y-2">
                                <li class="flex items-start">
                                    <i class="fas fa-bullseye text-gray-400 mt-1 mr-2 text-xs"></i>
                                    <span>Accuracy: 92.4% on test set</span>
                                </li>
                                <li class="flex items-start">
                                    <i class="fas fa-clock text-gray-400 mt-1 mr-2 text-xs"></i>
                                    <span>Average inference time: 120ms</span>
                                </li>
                                <li class="flex items-start">
                                    <i class="fas fa-database text-gray-400 mt-1 mr-2 text-xs"></i>
                                    <span>Trained on 15,000+ images</span>
                                </li>
                            </ul>
                        </div>
                        
                        <div class="bg-white rounded-lg p-4 shadow-sm">
                            <div class="flex items-center mb-3">
                                <div class="bg-purple-100 p-2 rounded-lg mr-3">
                                    <i class="fas fa-code-branch text-purple-600"></i>
                                </div>
                                <h3 class="font-medium">Development</h3>
                            </div>
                            <ul class="text-sm text-gray-600 space-y-2">
                                <li class="flex items-start">
                                    <i class="fab fa-python text-gray-400 mt-1 mr-2 text-xs"></i>
                                    <span>Built with PyTorch & Streamlit</span>
                                </li>
                                <li class="flex items-start">
                                    <i class="fab fa-github text-gray-400 mt-1 mr-2 text-xs"></i>
                                    <span>Open-source on GitHub</span>
                                </li>
                                <li class="flex items-start">
                                    <i class="fas fa-user text-gray-400 mt-1 mr-2 text-xs"></i>
                                    <span>By Hrishi Kamalachandran</span>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer class="bg-gray-800 text-white py-6 mt-12">
        <div class="container mx-auto px-4">
            <div class="flex flex-col md:flex-row justify-between items-center">
                <div class="mb-4 md:mb-0">
                    <div class="flex items-center space-x-2">
                        <i class="fas fa-eye text-xl"></i>
                        <span class="text-lg font-bold">OcuScan</span>
                    </div>
                    <p class="text-sm text-gray-400 mt-1">Advanced eye disease detection using deep learning</p>
                </div>
                <div class="flex space-x-6">
                    <a href="#" class="text-gray-300 hover:text-white">
                        <i class="fab fa-github text-xl"></i>
                    </a>
                    <a href="#" class="text-gray-300 hover:text-white">
                        <i class="fab fa-twitter text-xl"></i>
                    </a>
                    <a href="#" class="text-gray-300 hover:text-white">
                        <i class="fab fa-linkedin text-xl"></i>
                    </a>
                </div>
            </div>
            <div class="border-t border-gray-700 mt-6 pt-6 text-sm text-gray-400 text-center">
                <p>© 2025 OcuScan. This is a research prototype and not intended for clinical use.</p>
                <p class="mt-1">Developed by Hrishi Kamalachandran - hrishi.kamal16@gmail.com</p>
                <p class="mt-1">A Current High School Senior Interested In Pursuing The Medical Field</p>
            </div>
        </div>
    </footer>

    <script>
        // Sample data for demonstration
        const diseases = {
            'cataract': {
                description: 'Clouding of the natural lens of the eye, causing vision impairment. Cataracts typically develop slowly and can affect one or both eyes.',
                confidence: 87.5
            },
            'conjunctivitis': {
                description: 'Inflammation of the conjunctiva (pink eye), often caused by infection or allergies. Symptoms include redness, itching, and discharge.',
                confidence: 65.2
            },
            'swelling': {
                description: 'Eyelid or periorbital swelling, may indicate inflammation or infection. Can be caused by allergies, infections, or trauma.',
                confidence: 42.8
            },
            'normal': {
                description: 'No apparent pathological condition detected. The eye appears healthy with no signs of disease or abnormality.',
                confidence: 92.1
            },
            'uveitis': {
                description: 'Inflammation of the uvea, the middle layer of the eye. Can cause pain, redness, and vision problems if left untreated.',
                confidence: 38.7
            }
        };

        // DOM elements
        const uploadArea = document.getElementById('uploadArea');
        const uploadContent = document.getElementById('uploadContent');
        const fileInput = document.getElementById('fileInput');
        const imagePreview = document.getElementById('imagePreview');
        const previewImage = document.getElementById('previewImage');
        const changeImage = document.getElementById('changeImage');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const emptyState = document.getElementById('emptyState');
        const resultsSection = document.getElementById('resultsSection');
        const diagnosisResult = document.getElementById('diagnosisResult');
        const confidenceValue = document.getElementById('confidenceValue');
        const probabilityBars = document.getElementById('probabilityBars');
        const diseaseDescription = document.getElementById('diseaseDescription');

        // Event listeners
        uploadArea.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', handleFileSelect);
        
        changeImage.addEventListener('click', () => {
            fileInput.value = '';
            imagePreview.classList.add('hidden');
            uploadContent.classList.remove('hidden');
            analyzeBtn.classList.add('hidden');
            resultsSection.classList.add('hidden');
            emptyState.classList.remove('hidden');
        });
        
        analyzeBtn.addEventListener('click', analyzeImage);
        
        // Drag and drop functionality
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            uploadArea.classList.add('dragover');
        }
        
        function unhighlight() {
            uploadArea.classList.remove('dragover');
        }
        
        uploadArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length) {
                fileInput.files = files;
                handleFileSelect({ target: fileInput });
            }
        }
        
        // Handle file selection
        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            if (!file.type.match('image.*')) {
                alert('Please select an image file (JPG, PNG, BMP)');
                return;
            }
            
            if (file.size > 5 * 1024 * 1024) {
                alert('File size exceeds 5MB limit');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                uploadContent.classList.add('hidden');
                imagePreview.classList.remove('hidden');
                analyzeBtn.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        }
        
        // Analyze image (simulated)
        function analyzeImage() {
            // Show loading state
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Analyzing...';
            analyzeBtn.disabled = true;
            
            // Simulate API call with timeout
            setTimeout(() => {
                // Get a random disease for demo purposes
                const diseaseKeys = Object.keys(diseases);
                const randomDisease = diseaseKeys[Math.floor(Math.random() * diseaseKeys.length)];
                const disease = diseases[randomDisease];
                
                // Update UI with results
                diagnosisResult.textContent = randomDisease.charAt(0).toUpperCase() + randomDisease.slice(1);
                confidenceValue.textContent = disease.confidence.toFixed(1) + '%';
                diseaseDescription.textContent = disease.description;
                
                // Generate probability bars
                probabilityBars.innerHTML = '';
                
                // Create probabilities for all diseases (for demo)
                const allProbabilities = {};
                diseaseKeys.forEach(key => {
                    if (key === randomDisease) {
                        allProbabilities[key] = disease.confidence;
                    } else {
                        // Generate some random lower probabilities for other diseases
                        allProbabilities[key] = Math.random() * (100 - disease.confidence) * 0.5;
                    }
                });
                
                // Normalize to sum to 100%
                const total = Object.values(allProbabilities).reduce((sum, val) => sum + val, 0);
                Object.keys(allProbabilities).forEach(key => {
                    allProbabilities[key] = (allProbabilities[key] / total) * 100;
                });
                
                // Sort by probability
                const sorted = Object.entries(allProbabilities).sort((a, b) => b[1] - a[1]);
                
                // Create bars
                sorted.forEach(([diseaseName, probability]) => {
                    const bar = document.createElement('div');
                    bar.className = 'probability-item';
                    
                    const diseaseLabel = diseaseName.charAt(0).toUpperCase() + diseaseName.slice(1);
                    
                    bar.innerHTML = `
                        <div class="flex justify-between items-center mb-1">
                            <span class="text-sm font-medium text-gray-700">${diseaseLabel}</span>
                            <span class="text-sm font-medium text-gray-900">${probability.toFixed(1)}%</span>
                        </div>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${probability}%"></div>
                        </div>
                    `;
                    
                    probabilityBars.appendChild(bar);
                });
                
                // Show results
                emptyState.classList.add('hidden');
                resultsSection.classList.remove('hidden');
                
                // Reset button
                analyzeBtn.innerHTML = 'Analyze Image';
                analyzeBtn.disabled = false;
            }, 1500);
        }
    </script>
</body>
</html>
