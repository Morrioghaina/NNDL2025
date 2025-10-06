// Titanic Binary Classifier using TensorFlow.js
// Schema Definition - SWAP THESE FOR OTHER DATASETS
const TARGET_COLUMN = 'Survived';  // Binary target variable (0/1)
const FEATURE_COLUMNS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'];  // Feature columns
const ID_COLUMN = 'PassengerId';  // Identifier column (exclude from features)

// Global variables
let rawData = null;
let processedData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let testData = null;
let testPredictions = null;

// DOM elements
const elements = {
    loadDataBtn: document.getElementById('load-data-btn'),
    preprocessBtn: document.getElementById('preprocess-btn'),
    createModelBtn: document.getElementById('create-model-btn'),
    trainBtn: document.getElementById('train-btn'),
    evaluateBtn: document.getElementById('evaluate-btn'),
    predictBtn: document.getElementById('predict-btn'),
    exportModelBtn: document.getElementById('export-model-btn'),
    exportSubmissionBtn: document.getElementById('export-submission-btn'),
    exportProbabilitiesBtn: document.getElementById('export-probabilities-btn'),
    thresholdSlider: document.getElementById('threshold-slider'),
    thresholdValue: document.getElementById('threshold-value'),
    dataInfo: document.getElementById('data-info'),
    preprocessInfo: document.getElementById('preprocess-info'),
    modelSummary: document.getElementById('model-summary'),
    trainingProgress: document.getElementById('training-progress'),
    confusionMatrix: document.getElementById('confusion-matrix'),
    performanceMetrics: document.getElementById('performance-metrics'),
    predictionResults: document.getElementById('prediction-results')
};

// Initialize event listeners
function initializeEventListeners() {
    elements.loadDataBtn.addEventListener('click', loadData);
    elements.preprocessBtn.addEventListener('click', preprocessData);
    elements.createModelBtn.addEventListener('click', createModel);
    elements.trainBtn.addEventListener('click', trainModel);
    elements.evaluateBtn.addEventListener('click', evaluateModel);
    elements.predictBtn.addEventListener('click', predictTestData);
    elements.exportModelBtn.addEventListener('click', exportModel);
    elements.exportSubmissionBtn.addEventListener('click', exportSubmission);
    elements.exportProbabilitiesBtn.addEventListener('click', exportProbabilities);
    
    elements.thresholdSlider.addEventListener('input', updateThreshold);
}

// Load and parse CSV data
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile) {
        alert('Please upload training data (train.csv)');
        return;
    }
    
    try {
        elements.loadDataBtn.disabled = true;
        elements.loadDataBtn.textContent = 'Loading...';
        
        // Load training data
        const trainText = await readFile(trainFile);
        const trainData = parseCSV(trainText);
        
        // Load test data if provided
        if (testFile) {
            const testText = await readFile(testFile);
            testData = parseCSV(testText);
        }
        
        rawData = trainData;
        
        // Display data information
        displayDataInfo(trainData, testData);
        
        // Enable preprocessing button
        elements.preprocessBtn.disabled = false;
        
    } catch (error) {
        alert('Error loading data: ' + error.message);
        console.error(error);
    } finally {
        elements.loadDataBtn.disabled = false;
        elements.loadDataBtn.textContent = 'Load & Inspect Data';
    }
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('File reading failed'));
        reader.readAsText(file);
    });
}

// Parse CSV text to array of objects
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    return lines.slice(1).map(line => {
        const values = line.split(',').map(v => v.trim());
        const row = {};
        headers.forEach((header, index) => {
            let value = values[index];
            // Convert numeric values
            if (!isNaN(value) && value !== '') {
                value = parseFloat(value);
            }
            row[header] = value;
        });
        return row;
    });
}

// Display data information and visualizations
function displayDataInfo(trainData, testData) {
    let infoHTML = `<h3>Data Overview</h3>`;
    
    // Training data info
    infoHTML += `<p><strong>Training Data:</strong> ${trainData.length} rows, ${Object.keys(trainData[0]).length} columns</p>`;
    
    if (testData) {
        infoHTML += `<p><strong>Test Data:</strong> ${testData.length} rows, ${Object.keys(testData[0]).length} columns</p>`;
    }
    
    // Data preview
    infoHTML += `<h4>Data Preview (first 5 rows):</h4>`;
    infoHTML += `<table border="1" style="border-collapse: collapse; width: 100%;">`;
    infoHTML += `<tr>${Object.keys(trainData[0]).map(h => `<th>${h}</th>`).join('')}</tr>`;
    trainData.slice(0, 5).forEach(row => {
        infoHTML += `<tr>${Object.values(row).map(v => `<td>${v}</td>`).join('')}</tr>`;
    });
    infoHTML += `</table>`;
    
    // Missing values analysis
    infoHTML += `<h4>Missing Values (%):</h4>`;
    const missingInfo = calculateMissingValues(trainData);
    infoHTML += `<table border="1" style="border-collapse: collapse;">`;
    missingInfo.forEach(([col, missing]) => {
        infoHTML += `<tr><td>${col}</td><td>${missing.toFixed(2)}%</td></tr>`;
    });
    infoHTML += `</table>`;
    
    elements.dataInfo.innerHTML = infoHTML;
    
    // Create visualizations if tfvis is available
    if (typeof tfvis !== 'undefined') {
        createDataVisualizations(trainData);
    }
}

// Calculate missing values percentage
function calculateMissingValues(data) {
    const columns = Object.keys(data[0]);
    return columns.map(col => {
        const missingCount = data.filter(row => row[col] === '' || row[col] === null || row[col] === undefined || isNaN(row[col])).length;
        const missingPercent = (missingCount / data.length) * 100;
        return [col, missingPercent];
    });
}

// Create data visualizations using tfjs-vis
function createDataVisualizations(data) {
    // Survival by Sex
    const sexCounts = {
        'male': { survived: 0, died: 0 },
        'female': { survived: 0, died: 0 }
    };
    
    data.forEach(row => {
        if (row.Sex && row.Survived !== undefined && row.Survived !== '') {
            const sex = row.Sex.toLowerCase();
            const survived = parseInt(row.Survived);
            
            if (sexCounts[sex]) {
                if (survived === 1) {
                    sexCounts[sex].survived++;
                } else {
                    sexCounts[sex].died++;
                }
            }
        }
    });
    
    const sexData = [
        { index: 0, x: 'Male - Died', y: sexCounts.male.died },
        { index: 1, x: 'Male - Survived', y: sexCounts.male.survived },
        { index: 2, x: 'Female - Died', y: sexCounts.female.died },
        { index: 3, x: 'Female - Survived', y: sexCounts.female.survived }
    ];
    
    // Survival by Pclass
    const pclassCounts = {
        '1': { survived: 0, died: 0 },
        '2': { survived: 0, died: 0 },
        '3': { survived: 0, died: 0 }
    };
    
    data.forEach(row => {
        if (row.Pclass && row.Survived !== undefined && row.Survived !== '') {
            const pclass = row.Pclass.toString();
            const survived = parseInt(row.Survived);
            
            if (pclassCounts[pclass]) {
                if (survived === 1) {
                    pclassCounts[pclass].survived++;
                } else {
                    pclassCounts[pclass].died++;
                }
            }
        }
    });
    
    const pclassData = [
        { index: 0, x: 'Class 1 - Died', y: pclassCounts['1'].died },
        { index: 1, x: 'Class 1 - Survived', y: pclassCounts['1'].survived },
        { index: 2, x: 'Class 2 - Died', y: pclassCounts['2'].died },
        { index: 3, x: 'Class 2 - Survived', y: pclassCounts['2'].survived },
        { index: 4, x: 'Class 3 - Died', y: pclassCounts['3'].died },
        { index: 5, x: 'Class 3 - Survived', y: pclassCounts['3'].survived }
    ];
    
    // Create visualizations with proper configuration
    try {
        // Create a container for the visualizations
        const surfaceContainer = document.createElement('div');
        surfaceContainer.style.marginTop = '20px';
        elements.dataInfo.appendChild(surfaceContainer);
        
        // Render survival by sex
        tfvis.render.barchart(
            { name: 'Survival by Sex', tab: 'Data Inspection', container: surfaceContainer },
            sexData,
            {
                xLabel: 'Category',
                yLabel: 'Count',
                width: 400,
                height: 300
            }
        );
        
        // Render survival by passenger class
        tfvis.render.barchart(
            { name: 'Survival by Passenger Class', tab: 'Data Inspection', container: surfaceContainer },
            pclassData,
            {
                xLabel: 'Category',
                yLabel: 'Count',
                width: 500,
                height: 300
            }
        );
        
    } catch (error) {
        console.error('Error creating visualizations:', error);
        // Fallback: display data in table format
        displayFallbackVisualizations(sexData, pclassData);
    }
}

// Fallback display if tfvis fails
function displayFallbackVisualizations(sexData, pclassData) {
    let fallbackHTML = '<h4>Data Distribution (Fallback Display)</h4>';
    
    fallbackHTML += '<h5>Survival by Sex:</h5><table border="1" style="border-collapse: collapse; margin-bottom: 20px;">';
    fallbackHTML += '<tr><th>Category</th><th>Count</th></tr>';
    sexData.forEach(item => {
        fallbackHTML += `<tr><td>${item.x}</td><td>${item.y}</td></tr>`;
    });
    fallbackHTML += '</table>';
    
    fallbackHTML += '<h5>Survival by Passenger Class:</h5><table border="1" style="border-collapse: collapse;">';
    fallbackHTML += '<tr><th>Category</th><th>Count</th></tr>';
    pclassData.forEach(item => {
        fallbackHTML += `<tr><td>${item.x}</td><td>${item.y}</td></tr>`;
    });
    fallbackHTML += '</table>';
    
    const fallbackDiv = document.createElement('div');
    fallbackDiv.innerHTML = fallbackHTML;
    elements.dataInfo.appendChild(fallbackDiv);
}

// Preprocess the data
function preprocessData() {
    try {
        if (!rawData) {
            alert('Please load data first');
            return;
        }
        
        elements.preprocessBtn.disabled = true;
        elements.preprocessBtn.textContent = 'Preprocessing...';
        
        // Filter out rows with missing target
        const filteredData = rawData.filter(row => row[TARGET_COLUMN] !== undefined && row[TARGET_COLUMN] !== '');
        
        // Extract features and labels
        const features = [];
        const labels = [];
        const featureNames = new Set();
        
        filteredData.forEach(row => {
            const featureVector = extractFeatures(row);
            features.push(featureVector);
            labels.push(row[TARGET_COLUMN]);
            
            // Collect feature names
            Object.keys(featureVector).forEach(key => featureNames.add(key));
        });
        
        // Convert to tensors
        const featureArray = features.map(f => 
            Array.from(featureNames).map(name => f[name] || 0)
        );
        
        const featureTensor = tf.tensor2d(featureArray);
        const labelTensor = tf.tensor1d(labels);
        
        processedData = {
            features: featureTensor,
            labels: labelTensor,
            featureNames: Array.from(featureNames),
            rawFeatures: features,
            rawLabels: labels
        };
        
        displayPreprocessInfo(processedData);
        
        // Enable model creation
        elements.createModelBtn.disabled = false;
        
    } catch (error) {
        alert('Error preprocessing data: ' + error.message);
        console.error(error);
    } finally {
        elements.preprocessBtn.disabled = false;
        elements.preprocessBtn.textContent = 'Preprocess Data';
    }
}

// Extract features from a row
function extractFeatures(row) {
    const features = {};
    
    // Handle numeric features with imputation
    const age = row.Age === '' || row.Age === undefined || isNaN(row.Age) ? 29.7 : parseFloat(row.Age); // Median imputation
    const fare = row.Fare === '' || row.Fare === undefined || isNaN(row.Fare) ? 14.45 : parseFloat(row.Fare); // Median imputation
    
    // Standardize numeric features
    features.Age = (age - 29.7) / 13.0; // Rough standardization
    features.Fare = (fare - 14.45) / 50.0; // Rough standardization
    
    // One-hot encode categorical features
    features.Sex_male = row.Sex === 'male' ? 1 : 0;
    features.Sex_female = row.Sex === 'female' ? 1 : 0;
    
    features.Pclass_1 = row.Pclass === 1 ? 1 : 0;
    features.Pclass_2 = row.Pclass === 2 ? 1 : 0;
    features.Pclass_3 = row.Pclass === 3 ? 1 : 0;
    
    features.Embarked_C = row.Embarked === 'C' ? 1 : 0;
    features.Embarked_Q = row.Embarked === 'Q' ? 1 : 0;
    features.Embarked_S = row.Embarked === 'S' ? 1 : 0;
    
    // Optional features
    if (document.getElementById('family-size-toggle').checked) {
        const sibsp = row.SibSp || 0;
        const parch = row.Parch || 0;
        features.FamilySize = (sibsp + parch + 1) / 4.0; // Rough normalization
    }
    
    if (document.getElementById('is-alone-toggle').checked) {
        const sibsp = row.SibSp || 0;
        const parch = row.Parch || 0;
        features.IsAlone = (sibsp + parch === 0) ? 1 : 0;
    }
    
    return features;
}

// Display preprocessing information
function displayPreprocessInfo(processedData) {
    let infoHTML = `<h3>Preprocessing Results</h3>`;
    infoHTML += `<p><strong>Processed samples:</strong> ${processedData.features.shape[0]}</p>`;
    infoHTML += `<p><strong>Feature dimension:</strong> ${processedData.features.shape[1]}</p>`;
    infoHTML += `<p><strong>Feature names:</strong> ${processedData.featureNames.join(', ')}</p>`;
    
    elements.preprocessInfo.innerHTML = infoHTML;
}

// Create the neural network model
function createModel() {
    try {
        if (!processedData) {
            alert('Please preprocess data first');
            return;
        }
        
        const inputDim = processedData.features.shape[1];
        
        // Create sequential model
        model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputDim],
                    units: 16,
                    activation: 'relu',
                    name: 'hidden_layer'
                }),
                tf.layers.dense({
                    units: 1,
                    activation: 'sigmoid',
                    name: 'output_layer'
                })
            ]
        });
        
        // Compile the model
        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Display model summary
        displayModelSummary(model);
        
        // Enable training button
        elements.trainBtn.disabled = false;
        
    } catch (error) {
        alert('Error creating model: ' + error.message);
        console.error(error);
    }
}

// Display model summary
function displayModelSummary(model) {
    let summaryHTML = `<h3>Model Summary</h3>`;
    
    // Create simple summary since tfjs doesn't provide string summary in browser
    const totalParams = model.countParams();
    summaryHTML += `<p><strong>Architecture:</strong> Dense(16, relu) → Dense(1, sigmoid)</p>`;
    summaryHTML += `<p><strong>Total Parameters:</strong> ${totalParams.toLocaleString()}</p>`;
    summaryHTML += `<p><strong>Input Shape:</strong> [${model.inputs[0].shape.slice(1)}]</p>`;
    summaryHTML += `<p><strong>Output Shape:</strong> [${model.outputs[0].shape.slice(1)}]</p>`;
    summaryHTML += `<p><strong>Optimizer:</strong> Adam</p>`;
    summaryHTML += `<p><strong>Loss:</strong> binaryCrossentropy</p>`;
    summaryHTML += `<p><strong>Metrics:</strong> accuracy</p>`;
    
    elements.modelSummary.innerHTML = summaryHTML;
}

// Train the model
async function trainModel() {
    try {
        if (!model || !processedData) {
            alert('Please create model and preprocess data first');
            return;
        }
        
        elements.trainBtn.disabled = true;
        elements.trainBtn.textContent = 'Training...';
        
        // Split data into training and validation (80/20)
        const splitIndex = Math.floor(processedData.features.shape[0] * 0.8);
        
        const trainFeatures = processedData.features.slice(0, splitIndex);
        const trainLabels = processedData.labels.slice(0, splitIndex);
        
        const valFeatures = processedData.features.slice(splitIndex);
        const valLabels = processedData.labels.slice(splitIndex);
        
        validationData = {
            features: valFeatures,
            labels: valLabels,
            rawFeatures: processedData.rawFeatures.slice(splitIndex),
            rawLabels: processedData.rawLabels.slice(splitIndex)
        };
        
        // Train the model
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Training Performance' },
                ['loss', 'accuracy', 'val_loss', 'val_accuracy'],
                { callbacks: ['onEpochEnd'] }
            )
        });
        
        // Enable evaluation and prediction buttons
        elements.evaluateBtn.disabled = false;
        elements.predictBtn.disabled = false;
        elements.exportModelBtn.disabled = false;
        elements.thresholdSlider.disabled = false;
        
        elements.trainingProgress.innerHTML = '<p style="color: green;">Training completed successfully!</p>';
        
    } catch (error) {
        alert('Error training model: ' + error.message);
        console.error(error);
    } finally {
        elements.trainBtn.disabled = false;
        elements.trainBtn.textContent = 'Train Model';
    }
}

// Evaluate the model
async function evaluateModel() {
    try {
        if (!model || !validationData) {
            alert('Please train the model first');
            return;
        }
        
        // Get predictions
        const predictions = model.predict(validationData.features);
        const probs = await predictions.data();
        predictions.dispose();
        
        // Calculate metrics with current threshold
        const threshold = parseFloat(elements.thresholdSlider.value);
        updateMetrics(probs, validationData.rawLabels, threshold);
        
        // Plot ROC curve
        plotROCCurve(probs, validationData.rawLabels);
        
    } catch (error) {
        alert('Error evaluating model: ' + error.message);
        console.error(error);
    }
}

// Update metrics based on threshold
function updateMetrics(probabilities, trueLabels, threshold) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    probabilities.forEach((prob, i) => {
        const predicted = prob >= threshold ? 1 : 0;
        const actual = trueLabels[i];
        
        if (predicted === 1 && actual === 1) tp++;
        else if (predicted === 1 && actual === 0) fp++;
        else if (predicted === 0 && actual === 0) tn++;
        else if (predicted === 0 && actual === 1) fn++;
    });
    
    const accuracy = (tp + tn) / (tp + tn + fp + fn);
    const precision = tp === 0 ? 0 : tp / (tp + fp);
    const recall = tp === 0 ? 0 : tp / (tp + fn);
    const f1 = (precision + recall === 0) ? 0 : 2 * (precision * recall) / (precision + recall);
    
    // Update confusion matrix
    elements.confusionMatrix.innerHTML = `
        <table border="1" style="border-collapse: collapse; text-align: center;">
            <tr><th></th><th>Predicted 0</th><th>Predicted 1</th></tr>
            <tr><th>Actual 0</th><td>${tn}</td><td>${fp}</td></tr>
            <tr><th>Actual 1</th><td>${fn}</td><td>${tp}</td></tr>
        </table>
    `;
    
    // Update performance metrics
    elements.performanceMetrics.innerHTML = `
        <p><strong>Accuracy:</strong> ${accuracy.toFixed(4)}</p>
        <p><strong>Precision:</strong> ${precision.toFixed(4)}</p>
        <p><strong>Recall:</strong> ${recall.toFixed(4)}</p>
        <p><strong>F1-Score:</strong> ${f1.toFixed(4)}</p>
        <p><strong>Threshold:</strong> ${threshold.toFixed(2)}</p>
    `;
}

// Plot ROC curve
function plotROCCurve(probabilities, trueLabels) {
    // Calculate ROC points
    const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
    const rocPoints = [];
    
    thresholds.forEach(threshold => {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        probabilities.forEach((prob, i) => {
            const predicted = prob >= threshold ? 1 : 0;
            const actual = trueLabels[i];
            
            if (predicted === 1 && actual === 1) tp++;
            else if (predicted === 1 && actual === 0) fp++;
            else if (predicted === 0 && actual === 0) tn++;
            else if (predicted === 0 && actual === 1) fn++;
        });
        
        const tpr = tp === 0 ? 0 : tp / (tp + fn);
        const fpr = fp === 0 ? 0 : fp / (fp + tn);
        
        rocPoints.push({ x: fpr, y: tpr });
    });
    
    // Calculate AUC (trapezoidal rule)
    let auc = 0;
    for (let i = 1; i < rocPoints.length; i++) {
        const width = rocPoints[i].x - rocPoints[i-1].x;
        const avgHeight = (rocPoints[i].y + rocPoints[i-1].y) / 2;
        auc += width * avgHeight;
    }
    
    // Plot ROC curve with proper data structure
    try {
        const rocData = {
            values: rocPoints,
            series: ['ROC Curve']
        };
        
        tfvis.render.linechart(
            { name: `ROC Curve (AUC: ${auc.toFixed(4)})`, tab: 'Evaluation' },
            rocData,
            {
                xLabel: 'False Positive Rate',
                yLabel: 'True Positive Rate',
                xAxisDomain: [0, 1],
                yAxisDomain: [0, 1],
                width: 500,
                height: 400
            }
        );
        
        console.log(`ROC Curve plotted with AUC: ${auc.toFixed(4)}`);
        
    } catch (error) {
        console.error('Error plotting ROC curve:', error);
    }
}

// Update threshold from slider
function updateThreshold() {
    const threshold = parseFloat(elements.thresholdSlider.value);
    elements.thresholdValue.textContent = threshold.toFixed(2);
    
    if (validationData) {
        evaluateModel();
    }
}

// Predict on test data
async function predictTestData() {
    try {
        if (!model || !testData) {
            alert('Please load test data and train the model first');
            return;
        }
        
        elements.predictBtn.disabled = true;
        elements.predictBtn.textContent = 'Predicting...';
        
        // Preprocess test data
        const testFeatures = testData.map(row => extractFeatures(row));
        const featureNames = processedData.featureNames;
        
        const testFeatureArray = testFeatures.map(f => 
            featureNames.map(name => f[name] || 0)
        );
        
        const testFeatureTensor = tf.tensor2d(testFeatureArray);
        
        // Make predictions
        const predictions = model.predict(testFeatureTensor);
        const probabilities = await predictions.data();
        
        predictions.dispose();
        testFeatureTensor.dispose();
        
        testPredictions = probabilities;
        
        // Display prediction results
        displayPredictionResults(probabilities);
        
        // Enable export buttons
        elements.exportSubmissionBtn.disabled = false;
        elements.exportProbabilitiesBtn.disabled = false;
        
    } catch (error) {
        alert('Error making predictions: ' + error.message);
        console.error(error);
    } finally {
        elements.predictBtn.disabled = false;
        elements.predictBtn.textContent = 'Predict on Test Data';
    }
}

// Display prediction results
function displayPredictionResults(probabilities) {
    const threshold = parseFloat(elements.thresholdSlider.value);
    
    let resultsHTML = `<h3>Test Predictions</h3>`;
    resultsHTML += `<p><strong>Total predictions:</strong> ${probabilities.length}</p>`;
    
    const survivalCount = probabilities.filter(p => p >= threshold).length;
    resultsHTML += `<p><strong>Predicted survivors:</strong> ${survivalCount} (${(survivalCount/probabilities.length*100).toFixed(1)}%)</p>`;
    
    resultsHTML += `<h4>First 10 predictions:</h4>`;
    resultsHTML += `<table border="1" style="border-collapse: collapse;">`;
    resultsHTML += `<tr><th>PassengerId</th><th>Probability</th><th>Predicted</th></tr>`;
    
    probabilities.slice(0, 10).forEach((prob, i) => {
        const passengerId = testData[i] ? testData[i][ID_COLUMN] : `Test-${i+1}`;
        resultsHTML += `<tr>
            <td>${passengerId}</td>
            <td>${prob.toFixed(4)}</td>
            <td>${prob >= threshold ? 'Survived (1)' : 'Died (0)'}</td>
        </tr>`;
    });
    
    resultsHTML += `</table>`;
    
    elements.predictionResults.innerHTML = resultsHTML;
}

// Export model
async function exportModel() {
    try {
        if (!model) {
            alert('No model to export');
            return;
        }
        
        await model.save('downloads://titanic-tfjs-model');
        alert('Model exported successfully!');
        
    } catch (error) {
        alert('Error exporting model: ' + error.message);
        console.error(error);
    }
}

// Export submission CSV
function exportSubmission() {
    if (!testData || !testPredictions) {
        alert('No test predictions to export');
        return;
    }
    
    const threshold = parseFloat(elements.thresholdSlider.value);
    let csvContent = 'PassengerId,Survived\n';
    
    testData.forEach((row, i) => {
        const passengerId = row[ID_COLUMN];
        const survived = testPredictions[i] >= threshold ? 1 : 0;
        csvContent += `${passengerId},${survived}\n`;
    });
    
    downloadCSV(csvContent, 'titanic_submission.csv');
}

// Export probabilities CSV
function exportProbabilities() {
    if (!testData ||
