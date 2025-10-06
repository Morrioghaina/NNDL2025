// Titanic Binary Classifier with TensorFlow.js
// Dataset: https://www.kaggle.com/competitions/titanic/data

// Global variables to store data and model
let trainData = null;
let testData = null;
let mergedData = null;
let model = null;
let trainingHistory = null;
let testPredictions = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;

// Data Schema - Modify these for other datasets
const TARGET_COLUMN = 'Survived';
const FEATURE_COLUMNS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'];
const ID_COLUMN = 'PassengerId';

// DOM elements
const loadDataBtn = document.getElementById('loadDataBtn');
const preprocessBtn = document.getElementById('preprocessBtn');
const createModelBtn = document.getElementById('createModelBtn');
const trainBtn = document.getElementById('trainBtn');
const predictBtn = document.getElementById('predictBtn');
const exportBtn = document.getElementById('exportBtn');
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdValue = document.getElementById('thresholdValue');

// Event listeners
loadDataBtn.addEventListener('click', loadData);
preprocessBtn.addEventListener('click', preprocessData);
createModelBtn.addEventListener('click', createModel);
trainBtn.addEventListener('click', trainModel);
predictBtn.addEventListener('click', predictTestSet);
exportBtn.addEventListener('click', exportResults);
thresholdSlider.addEventListener('input', updateThreshold);

// Load and parse CSV data
async function loadData() {
    const trainFile = document.getElementById('trainFile').files[0];
    const testFile = document.getElementById('testFile').files[0];
    
    if (!trainFile || !testFile) {
        alert('Please upload both train.csv and test.csv files');
        return;
    }
    
    try {
        // Parse train data
        const trainText = await readFile(trainFile);
        const trainResult = Papa.parse(trainText, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true
        });
        
        if (trainResult.errors.length > 0) {
            console.error('Train CSV parsing errors:', trainResult.errors);
            alert('Error parsing train.csv. Check console for details.');
            return;
        }
        
        trainData = trainResult.data;
        
        // Parse test data
        const testText = await readFile(testFile);
        const testResult = Papa.parse(testText, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true
        });
        
        if (testResult.errors.length > 0) {
            console.error('Test CSV parsing errors:', testResult.errors);
            alert('Error parsing test.csv. Check console for details.');
            return;
        }
        
        testData = testResult.data;
        
        // Add source column to distinguish train/test
        trainData.forEach(row => row.source = 'train');
        testData.forEach(row => row.source = 'test');
        
        // Merge datasets for analysis
        mergedData = [...trainData, ...testData];
        
        // Display data info
        displayDataInfo();
        
        // Enable preprocessing button
        preprocessBtn.disabled = false;
        
    } catch (error) {
        console.error('Error loading data:', error);
        alert('Error loading data: ' + error.message);
    }
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsText(file);
    });
}

// Display data information
function displayDataInfo() {
    const dataInfo = document.getElementById('dataInfo');
    
    let html = `
        <h3>Data Overview</h3>
        <p>Train samples: ${trainData.length}</p>
        <p>Test samples: ${testData.length}</p>
        <p>Total samples: ${mergedData.length}</p>
        
        <h4>First 5 rows of training data:</h4>
        <table>
            <thead>
                <tr>
                    ${Object.keys(trainData[0]).map(key => `<th>${key}</th>`).join('')}
                </tr>
            </thead>
            <tbody>
                ${trainData.slice(0, 5).map(row => `
                    <tr>
                        ${Object.values(row).map(val => `<td>${val}</td>`).join('')}
                    </tr>
                `).join('')}
            </tbody>
        </table>
        
        <h4>Missing Values:</h4>
        <table>
            <thead>
                <tr>
                    <th>Column</th>
                    <th>Missing Count</th>
                    <th>Missing %</th>
                </tr>
            </thead>
            <tbody>
                ${Object.keys(trainData[0]).map(col => {
                    const missingCount = trainData.filter(row => row[col] === null || row[col] === undefined || row[col] === '').length;
                    const missingPercent = ((missingCount / trainData.length) * 100).toFixed(2);
                    return `
                        <tr>
                            <td>${col}</td>
                            <td>${missingCount}</td>
                            <td>${missingPercent}%</td>
                        </tr>
                    `;
                }).join('')}
            </tbody>
        </table>
    `;
    
    dataInfo.innerHTML = html;
    
    // Show survival charts
    showSurvivalCharts();
}

// Show survival charts using tfjs-vis
function showSurvivalCharts() {
    // Survival by Sex
    const sexSurvival = {};
    trainData.forEach(row => {
        if (row.Sex && row.Survived !== undefined) {
            if (!sexSurvival[row.Sex]) {
                sexSurvival[row.Sex] = { survived: 0, total: 0 };
            }
            sexSurvival[row.Sex].total++;
            if (row.Survived === 1) {
                sexSurvival[row.Sex].survived++;
            }
        }
    });
    
    const sexData = Object.keys(sexSurvival).map(sex => ({
        x: sex,
        y: (sexSurvival[sex].survived / sexSurvival[sex].total) * 100
    }));
    
    // Survival by Pclass
    const pclassSurvival = {};
    trainData.forEach(row => {
        if (row.Pclass && row.Survived !== undefined) {
            if (!pclassSurvival[row.Pclass]) {
                pclassSurvival[row.Pclass] = { survived: 0, total: 0 };
            }
            pclassSurvival[row.Pclass].total++;
            if (row.Survived === 1) {
                pclassSurvival[row.Pclass].survived++;
            }
        }
    });
    
    const pclassData = Object.keys(pclassSurvival).map(pclass => ({
        x: `Class ${pclass}`,
        y: (pclassSurvival[pclass].survived / pclassSurvival[pclass].total) * 100
    }));
    
    // Create charts
    const surface = { name: 'Survival Analysis', tab: 'Charts' };
    
    tfvis.render.barchart(surface, sexData, {
        xLabel: 'Sex',
        yLabel: 'Survival Rate (%)',
        width: 400,
        height: 300
    });
    
    const surface2 = { name: 'Survival by Class', tab: 'Charts' };
    
    tfvis.render.barchart(surface2, pclassData, {
        xLabel: 'Passenger Class',
        yLabel: 'Survival Rate (%)',
        width: 400,
        height: 300
    });
}

// Preprocess the data
function preprocessData() {
    if (!trainData || !testData) {
        alert('Please load data first');
        return;
    }
    
    try {
        // Calculate median Age from train data
        const ages = trainData.map(row => row.Age).filter(age => age !== null && age !== undefined);
        const medianAge = ages.length > 0 ? 
            ages.sort((a, b) => a - b)[Math.floor(ages.length / 2)] : 0;
        
        // Calculate mode Embarked from train data
        const embarkedCounts = {};
        trainData.forEach(row => {
            if (row.Embarked) {
                embarkedCounts[row.Embarked] = (embarkedCounts[row.Embarked] || 0) + 1;
            }
        });
        const modeEmbarked = Object.keys(embarkedCounts).reduce((a, b) => 
            embarkedCounts[a] > embarkedCounts[b] ? a : b, 'S');
        
        // Calculate mean and std for Fare from train data
        const fares = trainData.map(row => row.Fare).filter(fare => fare !== null && fare !== undefined);
        const meanFare = fares.reduce((sum, fare) => sum + fare, 0) / fares.length;
        const stdFare = Math.sqrt(
            fares.reduce((sum, fare) => sum + Math.pow(fare - meanFare, 2), 0) / fares.length
        );
        
        // Preprocess function
        const preprocessRow = (row, isTrain = true) => {
            // Impute missing values
            const age = row.Age !== null && row.Age !== undefined ? row.Age : medianAge;
            const embarked = row.Embarked || modeEmbarked;
            let fare = row.Fare !== null && row.Fare !== undefined ? row.Fare : meanFare;
            
            // Standardize Fare
            fare = (fare - meanFare) / stdFare;
            
            // Create additional features
            const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
            const isAlone = familySize === 1 ? 1 : 0;
            
            // One-hot encoding for categorical variables
            const sex = row.Sex === 'female' ? 1 : 0;
            const pclass1 = row.Pclass === 1 ? 1 : 0;
            const pclass2 = row.Pclass === 2 ? 1 : 0;
            const pclass3 = row.Pclass === 3 ? 1 : 0;
            const embarkedC = embarked === 'C' ? 1 : 0;
            const embarkedQ = embarked === 'Q' ? 1 : 0;
            const embarkedS = embarked === 'S' ? 1 : 0;
            
            // Create feature array
            const features = [
                pclass1, pclass2, pclass3, // Pclass (one-hot)
                sex,                       // Sex (binary)
                age,                       // Age
                row.SibSp || 0,            // SibSp
                row.Parch || 0,            // Parch
                fare,                      // Fare (standardized)
                embarkedC, embarkedQ, embarkedS, // Embarked (one-hot)
                familySize,                // FamilySize
                isAlone                    // IsAlone
            ];
            
            return {
                features,
                label: isTrain ? row.Survived : null,
                passengerId: row.PassengerId
            };
        };
        
        // Preprocess all data
        const preprocessedTrain = trainData.map(row => preprocessRow(row, true));
        const preprocessedTest = testData.map(row => preprocessRow(row, false));
        
        // Filter out rows with null labels in training data
        const validTrainData = preprocessedTrain.filter(item => item.label !== null && item.label !== undefined);
        
        // Convert to tensors
        const trainFeatures = validTrainData.map(item => item.features);
        const trainLabels = validTrainData.map(item => item.label);
        
        // Store preprocessed data
        window.preprocessedData = {
            trainFeatures,
            trainLabels,
            preprocessedTest,
            featureNames: [
                'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex', 'Age', 'SibSp', 'Parch', 
                'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'FamilySize', 'IsAlone'
            ]
        };
        
        // Display preprocessing info
        const preprocessingInfo = document.getElementById('preprocessingInfo');
        preprocessingInfo.innerHTML = `
            <h3>Preprocessing Complete</h3>
            <p>Training samples: ${trainFeatures.length}</p>
            <p>Test samples: ${preprocessedTest.length}</p>
            <p>Features: ${window.preprocessedData.featureNames.join(', ')}</p>
            <p>Imputed Age (median): ${medianAge.toFixed(2)}</p>
            <p>Imputed Embarked (mode): ${modeEmbarked}</p>
            <p>Standardized Fare (mean: ${meanFare.toFixed(2)}, std: ${stdFare.toFixed(2)})</p>
        `;
        
        // Enable model creation button
        createModelBtn.disabled = false;
        
    } catch (error) {
        console.error('Error preprocessing data:', error);
        alert('Error preprocessing data: ' + error.message);
    }
}

// Create the model
function createModel() {
    if (!window.preprocessedData) {
        alert('Please preprocess data first');
        return;
    }
    
    try {
        // Create sequential model
        model = tf.sequential();
        
        // Add hidden layer
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            inputShape: [window.preprocessedData.trainFeatures[0].length]
        }));
        
        // Add output layer
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));
        
        // Compile the model
        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Display model info
        const modelInfo = document.getElementById('modelInfo');
        modelInfo.innerHTML = `
            <h3>Model Created</h3>
            <p>Architecture: Input(${window.preprocessedData.trainFeatures[0].length}) → Dense(16, relu) → Dense(1, sigmoid)</p>
            <p>Optimizer: Adam</p>
            <p>Loss: binaryCrossentropy</p>
            <p>Metrics: accuracy</p>
        `;
        
        // Print model summary to console
        model.summary();
        
        // Enable training button
        trainBtn.disabled = false;
        
    } catch (error) {
        console.error('Error creating model:', error);
        alert('Error creating model: ' + error.message);
    }
}

// Train the model
async function trainModel() {
    if (!model || !window.preprocessedData) {
        alert('Please create model first');
        return;
    }
    
    try {
        const { trainFeatures, trainLabels } = window.preprocessedData;
        
        // Convert to tensors
        const featuresTensor = tf.tensor2d(trainFeatures);
        const labelsTensor = tf.tensor1d(trainLabels);
        
        // Create validation split (80/20)
        const splitIndex = Math.floor(trainFeatures.length * 0.8);
        
        const trainFeaturesTensor = featuresTensor.slice([0, 0], [splitIndex, -1]);
        const trainLabelsTensor = labelsTensor.slice([0], [splitIndex]);
        
        const valFeaturesTensor = featuresTensor.slice([splitIndex, 0], [-1, -1]);
        const valLabelsTensor = labelsTensor.slice([splitIndex], [-1]);
        
        // Store validation data for later evaluation
        validationData = valFeaturesTensor;
        validationLabels = valLabelsTensor;
        
        // Training parameters
        const epochs = 50;
        const batchSize = 32;
        
        // Train the model with tfvis callbacks
        trainingHistory = await model.fit(trainFeaturesTensor, trainLabelsTensor, {
            epochs,
            batchSize,
            validationData: [valFeaturesTensor, valLabelsTensor],
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Training Performance' },
                ['loss', 'val_loss', 'acc', 'val_acc'],
                { callbacks: ['onEpochEnd'] }
            )
        });
        
        // Make predictions on validation set
        validationPredictions = model.predict(valFeaturesTensor);
        
        // Enable prediction and metrics
        predictBtn.disabled = false;
        thresholdSlider.disabled = false;
        
        // Display training info
        const trainingInfo = document.getElementById('trainingInfo');
        const finalLoss = trainingHistory.history.loss[trainingHistory.history.loss.length - 1];
        const finalAcc = trainingHistory.history.acc[trainingHistory.history.acc.length - 1];
        const finalValLoss = trainingHistory.history.val_loss[trainingHistory.history.val_loss.length - 1];
        const finalValAcc = trainingHistory.history.val_acc[trainingHistory.history.val_acc.length - 1];
        
        trainingInfo.innerHTML = `
            <h3>Training Complete</h3>
            <p>Final Training Loss: ${finalLoss.toFixed(4)}</p>
            <p>Final Training Accuracy: ${finalAcc.toFixed(4)}</p>
            <p>Final Validation Loss: ${finalValLoss.toFixed(4)}</p>
            <p>Final Validation Accuracy: ${finalValAcc.toFixed(4)}</p>
        `;
        
        // Calculate and display initial metrics
        updateThreshold();
        
        // Clean up tensors
        featuresTensor.dispose();
        labelsTensor.dispose();
        trainFeaturesTensor.dispose();
        trainLabelsTensor.dispose();
        
    } catch (error) {
        console.error('Error training model:', error);
        alert('Error training model: ' + error.message);
    }
}

// Update metrics based on threshold
async function updateThreshold() {
    if (!validationPredictions || !validationLabels) return;
    
    const threshold = parseFloat(thresholdSlider.value);
    thresholdValue.textContent = threshold.toFixed(2);
    
    try {
        // Convert predictions to binary using threshold
        const binaryPredictions = validationPredictions.greater(tf.scalar(threshold));
        
        // Calculate confusion matrix
        const confusionMatrix = await tfvis.metrics.confusionMatrix(
            validationLabels, 
            binaryPredictions
        );
        
        // Calculate metrics
        const tn = confusionMatrix[0][0];
        const fp = confusionMatrix[0][1];
        const fn = confusionMatrix[1][0];
        const tp = confusionMatrix[1][1];
        
        const accuracy = (tp + tn) / (tp + tn + fp + fn);
        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1 = 2 * (precision * recall) / (precision + recall) || 0;
        
        // Display metrics
        const metricsInfo = document.getElementById('metricsInfo');
        metricsInfo.innerHTML = `
            <div class="metrics-container">
                <div class="metric-box">
                    <h4>Confusion Matrix</h4>
                    <p>True Negative: ${tn}</p>
                    <p>False Positive: ${fp}</p>
                    <p>False Negative: ${fn}</p>
                    <p>True Positive: ${tp}</p>
                </div>
                <div class="metric-box">
                    <h4>Performance Metrics</h4>
                    <p>Accuracy: ${accuracy.toFixed(4)}</p>
                    <p>Precision: ${precision.toFixed(4)}</p>
                    <p>Recall: ${recall.toFixed(4)}</p>
                    <p>F1-Score: ${f1.toFixed(4)}</p>
                </div>
            </div>
        `;
        
        // Display evaluation table
        const evaluationTable = document.getElementById('evaluationTable');
        evaluationTable.innerHTML = `
            <h4>Evaluation Metrics</h4>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Threshold</td>
                        <td>${threshold.toFixed(2)}</td>
                    </tr>
                    <tr>
                        <td>Accuracy</td>
                        <td>${accuracy.toFixed(4)}</td>
                    </tr>
                    <tr>
                        <td>Precision</td>
                        <td>${precision.toFixed(4)}</td>
                    </tr>
                    <tr>
                        <td>Recall</td>
                        <td>${recall.toFixed(4)}</td>
                    </tr>
                    <tr>
                        <td>F1-Score</td>
                        <td>${f1.toFixed(4)}</td>
                    </tr>
                    <tr>
                        <td>True Positives</td>
                        <td>${tp}</td>
                    </tr>
                    <tr>
                        <td>False Positives</td>
                        <td>${fp}</td>
                    </tr>
                    <tr>
                        <td>True Negatives</td>
                        <td>${tn}</td>
                    </tr>
                    <tr>
                        <td>False Negatives</td>
                        <td>${fn}</td>
                    </tr>
                </tbody>
            </table>
        `;
        
        // Plot ROC curve
        plotROC();
        
    } catch (error) {
        console.error('Error updating metrics:', error);
    }
}

// Plot ROC curve
async function plotROC() {
    if (!validationPredictions || !validationLabels) return;
    
    try {
        // Get prediction probabilities and true labels as arrays
        const probs = await validationPredictions.data();
        const labels = await validationLabels.data();
        
        // Calculate ROC curve points
        const thresholds = Array.from({ length: 100 }, (_, i) => i / 100);
        const rocPoints = [];
        
        for (const threshold of thresholds) {
            let tp = 0, fp = 0, tn = 0, fn = 0;
            
            for (let i = 0; i < probs.length; i++) {
                const prediction = probs[i] >= threshold ? 1 : 0;
                const actual = labels[i];
                
                if (prediction === 1 && actual === 1) tp++;
                else if (prediction === 1 && actual === 0) fp++;
                else if (prediction === 0 && actual === 0) tn++;
                else if (prediction === 0 && actual === 1) fn++;
            }
            
            const tpr = tp / (tp + fn) || 0;
            const fpr = fp / (fp + tn) || 0;
            
            rocPoints.push({ x: fpr, y: tpr, threshold });
        }
        
        // Calculate AUC (approximate using trapezoidal rule)
        let auc = 0;
        for (let i = 1; i < rocPoints.length; i++) {
            auc += (rocPoints[i].x - rocPoints[i-1].x) * 
                   (rocPoints[i].y + rocPoints[i-1].y) / 2;
        }
        
        // Display AUC
        const metricsInfo = document.getElementById('metricsInfo');
        metricsInfo.innerHTML += `
            <div class="metric-box">
                <h4>ROC Curve</h4>
                <p>AUC: ${auc.toFixed(4)}</p>
            </div>
        `;
        
        // Plot ROC curve
        const surface = { name: 'ROC Curve', tab: 'Evaluation' };
        tfvis.render.linechart(surface, {
            values: [rocPoints],
            series: ['ROC Curve']
        }, {
            xLabel: 'False Positive Rate',
            yLabel: 'True Positive Rate',
            width: 400,
            height: 400
        });
        
    } catch (error) {
        console.error('Error plotting ROC:', error);
    }
}

// Predict on test set
async function predictTestSet() {
    if (!model || !window.preprocessedData) {
        alert('Please train model first');
        return;
    }
    
    try {
        const { preprocessedTest } = window.preprocessedData;
        
        // Extract features from test data
        const testFeatures = preprocessedTest.map(item => item.features);
        const testFeaturesTensor = tf.tensor2d(testFeatures);
        
        // Make predictions
        const predictionsTensor = model.predict(testFeaturesTensor);
        const predictions = await predictionsTensor.data();
        
        // Store predictions
        testPredictions = preprocessedTest.map((item, index) => ({
            passengerId: item.passengerId,
            probability: predictions[index],
            survived: predictions[index] >= parseFloat(thresholdSlider.value) ? 1 : 0
        }));
        
        // Display prediction info
        const predictionInfo = document.getElementById('predictionInfo');
        predictionInfo.innerHTML = `
            <h3>Predictions Complete</h3>
            <p>Predicted survival for ${testPredictions.length} test samples</p>
            <p>Survival rate: ${(testPredictions.filter(p => p.survived === 1).length / testPredictions.length * 100).toFixed(2)}%</p>
            
            <h4>First 10 predictions:</h4>
            <table>
                <thead>
                    <tr>
                        <th>PassengerId</th>
                        <th>Probability</th>
                        <th>Survived</th>
                    </tr>
                </thead>
                <tbody>
                    ${testPredictions.slice(0, 10).map(p => `
                        <tr>
                            <td>${p.passengerId}</td>
                            <td>${p.probability.toFixed(4)}</td>
                            <td>${p.survived}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        
        // Enable export button
        exportBtn.disabled = false;
        
        // Clean up tensor
        testFeaturesTensor.dispose();
        predictionsTensor.dispose();
        
    } catch (error) {
        console.error('Error predicting test set:', error);
        alert('Error predicting test set: ' + error.message);
    }
}

// Export results
function exportResults() {
    if (!testPredictions) {
        alert('No predictions to export');
        return;
    }
    
    try {
        // Create submission CSV (PassengerId, Survived)
        const submissionCsv = Papa.unparse({
            fields: ['PassengerId', 'Survived'],
            data: testPredictions.map(p => [p.passengerId, p.survived])
        });
        
        // Create probabilities CSV (PassengerId, Probability)
        const probabilitiesCsv = Papa.unparse({
            fields: ['PassengerId', 'Survived_Probability'],
            data: testPredictions.map(p => [p.passengerId, p.probability])
        });
        
        // Create download links
        downloadFile(submissionCsv, 'submission.csv', 'text/csv');
        downloadFile(probabilitiesCsv, 'probabilities.csv', 'text/csv');
        
        // Save model (this will trigger a download)
        model.save('downloads://titanic-tfjs-model');
        
        // Display export info
        const exportInfo = document.getElementById('exportInfo');
        exportInfo.innerHTML = `
            <h3>Export Complete</h3>
            <p>Downloaded submission.csv with binary predictions</p>
            <p>Downloaded probabilities.csv with prediction probabilities</p>
            <p>Downloaded model files (model.json and weights.bin)</p>
        `;
        
    } catch (error) {
        console.error('Error exporting results:', error);
        alert('Error exporting results: ' + error.message);
    }
}

// Helper function to download files
function downloadFile(content, filename, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Note for reusability: To use with other datasets, modify:
// 1. TARGET_COLUMN, FEATURE_COLUMNS, ID_COLUMN constants
// 2. The preprocessRow function to handle your specific feature engineering
// 3. The model architecture if needed for different input dimensions
