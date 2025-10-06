// Titanic Binary Classifier using TensorFlow.js
// Running entirely in browser - no server needed

// ========================================
// DATA SCHEMA CONFIGURATION
// ========================================
// REUSE NOTE: To adapt for other datasets, modify this schema section
const SCHEMA = {
    target: 'Survived',           // Target column name (0/1 for binary classification)
    identifier: 'PassengerId',    // ID column to exclude from features
    features: {
        numerical: ['Age', 'SibSp', 'Parch', 'Fare'],      // Continuous features
        categorical: ['Pclass', 'Sex', 'Embarked']         // Categorical features
    }
};

// ========================================
// GLOBAL VARIABLES
// ========================================
let trainData = null;
let testData = null;
let processedTrain = null;
let processedTest = null;
let model = null;
let trainFeatures = null;
let trainLabels = null;
let valFeatures = null;
let valLabels = null;
let testPredictions = null;
let scaler = {};
let encoderMappings = {};
let featureNames = [];

// ========================================
// UTILITY FUNCTIONS
// ========================================

// Parse CSV content
function parseCSV(content) {
    const lines = content.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        const row = {};
        headers.forEach((header, index) => {
            let value = values[index]?.trim();
            // Handle numeric values
            if (!isNaN(value) && value !== '') {
                row[header] = parseFloat(value);
            } else {
                row[header] = value || null;
            }
        });
        data.push(row);
    }
    
    return { headers, data };
}

// Calculate statistics for data inspection
function calculateStats(data, columns) {
    const stats = {};
    
    columns.forEach(col => {
        const values = data.map(row => row[col]).filter(v => v !== null && v !== '');
        const missing = data.length - values.length;
        
        stats[col] = {
            missing: missing,
            missingPct: ((missing / data.length) * 100).toFixed(1),
            unique: [...new Set(values)].length
        };
        
        // For numerical columns
        const numValues = values.filter(v => !isNaN(v)).map(v => parseFloat(v));
        if (numValues.length > 0) {
            stats[col].mean = (numValues.reduce((a, b) => a + b, 0) / numValues.length).toFixed(2);
            stats[col].median = getMedian(numValues).toFixed(2);
            stats[col].min = Math.min(...numValues).toFixed(2);
            stats[col].max = Math.max(...numValues).toFixed(2);
        }
    });
    
    return stats;
}

// Get median value
function getMedian(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Get mode (most frequent value)
function getMode(values) {
    const counts = {};
    values.forEach(v => {
        if (v !== null && v !== '') {
            counts[v] = (counts[v] || 0) + 1;
        }
    });
    
    let maxCount = 0;
    let mode = null;
    
    for (const [value, count] of Object.entries(counts)) {
        if (count > maxCount) {
            maxCount = count;
            mode = value;
        }
    }
    
    return mode;
}

// ========================================
// DATA LOADING AND INSPECTION
// ========================================

// Handle file input changes
document.getElementById('train-file').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById('train-label').classList.add('loaded');
        document.getElementById('train-label').textContent = `✅ ${file.name}`;
    }
});

document.getElementById('test-file').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById('test-label').classList.add('loaded');
        document.getElementById('test-label').textContent = `✅ ${file.name}`;
    }
});

// Load and inspect data
document.getElementById('load-data-btn').addEventListener('click', async function() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile || !testFile) {
        alert('Please select both train.csv and test.csv files');
        return;
    }
    
    try {
        // Read files
        const trainContent = await trainFile.text();
        const testContent = await testFile.text();
        
        // Parse CSV
        const trainParsed = parseCSV(trainContent);
        const testParsed = parseCSV(testContent);
        
        trainData = trainParsed.data;
        testData = testParsed.data;
        
        // Display preview
        displayDataPreview(trainData.slice(0, 5), 'Training Data Preview');
        
        // Display statistics
        displayDataStats(trainData, trainParsed.headers);
        
        // Create visualizations
        createDataVisualizations(trainData);
        
        // Update UI
        document.getElementById('data-status').classList.add('active');
        document.getElementById('preprocess-btn').disabled = false;
        
        showSuccess('Data loaded successfully! ' + 
                   `Train: ${trainData.length} rows, Test: ${testData.length} rows`);
        
    } catch (error) {
        alert('Error loading data: ' + error.message);
        console.error(error);
    }
});

// Display data preview table
function displayDataPreview(data, title) {
    const previewDiv = document.getElementById('data-preview');
    let html = `<h3>${title}</h3><div class="preview-table"><table>`;
    
    // Headers
    const headers = Object.keys(data[0]);
    html += '<tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr>';
    
    // Rows
    data.forEach(row => {
        html += '<tr>';
        headers.forEach(header => {
            const value = row[header];
            html += `<td>${value !== null ? value : '<em>null</em>'}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</table></div>';
    previewDiv.innerHTML = html;
}

// Display data statistics
function displayDataStats(data, headers) {
    const statsDiv = document.getElementById('data-stats');
    const stats = calculateStats(data, headers);
    
    let html = '<h3>Data Statistics</h3><div class="metrics-grid">';
    
    for (const [col, colStats] of Object.entries(stats)) {
        html += `
            <div class="metric-card">
                <div style="font-weight: bold; color: #667eea; margin-bottom: 10px;">${col}</div>
                <div style="font-size: 0.9em; text-align: left;">
                    Missing: ${colStats.missingPct}%<br>
                    Unique: ${colStats.unique}<br>
                    ${colStats.mean ? `Mean: ${colStats.mean}<br>` : ''}
                    ${colStats.median ? `Median: ${colStats.median}<br>` : ''}
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    statsDiv.innerHTML = html;
}

// Create data visualizations using tfjs-vis
function createDataVisualizations(data) {
    // Survival by Sex
    const sexSurvival = {};
    data.forEach(row => {
        if (row.Sex && row.Survived !== null) {
            if (!sexSurvival[row.Sex]) {
                sexSurvival[row.Sex] = { survived: 0, died: 0 };
            }
            if (row.Survived === 1) {
                sexSurvival[row.Sex].survived++;
            } else {
                sexSurvival[row.Sex].died++;
            }
        }
    });
    
    const sexData = Object.keys(sexSurvival).map(sex => ({
        index: sex,
        value: (sexSurvival[sex].survived / (sexSurvival[sex].survived + sexSurvival[sex].died)) * 100
    }));
    
    // Survival by Pclass
    const pclassSurvival = {};
    data.forEach(row => {
        if (row.Pclass && row.Survived !== null) {
            const pclass = `Class ${row.Pclass}`;
            if (!pclassSurvival[pclass]) {
                pclassSurvival[pclass] = { survived: 0, died: 0 };
            }
            if (row.Survived === 1) {
                pclassSurvival[pclass].survived++;
            } else {
                pclassSurvival[pclass].died++;
            }
        }
    });
    
    const pclassData = Object.keys(pclassSurvival).sort().map(pclass => ({
        index: pclass,
        value: (pclassSurvival[pclass].survived / 
               (pclassSurvival[pclass].survived + pclassSurvival[pclass].died)) * 100
    }));
    
    // Create visualizations
    const vizContainer = document.getElementById('data-stats');
    const vizDiv = document.createElement('div');
    vizDiv.innerHTML = '<h3>Survival Rate Analysis</h3>';
    vizDiv.style.display = 'flex';
    vizDiv.style.gap = '20px';
    vizDiv.style.marginTop = '20px';
    
    const sexViz = document.createElement('div');
    sexViz.style.flex = '1';
    vizDiv.appendChild(sexViz);
    
    const pclassViz = document.createElement('div');
    pclassViz.style.flex = '1';
    vizDiv.appendChild(pclassViz);
    
    vizContainer.appendChild(vizDiv);
    
    // Render charts
    tfvis.render.barchart(
        sexViz,
        sexData,
        { width: 300, height: 200, xLabel: 'Sex', yLabel: 'Survival Rate (%)', fontSize: 12 }
    );
    
    tfvis.render.barchart(
        pclassViz,
        pclassData,
        { width: 300, height: 200, xLabel: 'Class', yLabel: 'Survival Rate (%)', fontSize: 12 }
    );
}

// ========================================
// DATA PREPROCESSING
// ========================================

document.getElementById('preprocess-btn').addEventListener('click', function() {
    try {
        // Preprocess training data
        processedTrain = preprocessData(trainData, true);
        
        // Preprocess test data using training statistics
        processedTest = preprocessData(testData, false);
        
        // Display preprocessing info
        displayPreprocessingInfo();
        
        // Update UI
        document.getElementById('preprocess-status').classList.add('active');
        document.getElementById('create-model-btn').disabled = false;
        
        showSuccess('Data preprocessing completed!');
        
    } catch (error) {
        alert('Error preprocessing data: ' + error.message);
        console.error(error);
    }
});

function preprocessData(data, fitScaler = true) {
    const processed = JSON.parse(JSON.stringify(data)); // Deep copy
    
    // Imputation
    // Age - use median
    const ageValues = data.map(row => row.Age).filter(v => v !== null && !isNaN(v));
    const ageMedian = fitScaler ? getMedian(ageValues) : scaler.ageMedian;
    if (fitScaler) scaler.ageMedian = ageMedian;
    
    // Embarked - use mode
    const embarkedValues = data.map(row => row.Embarked).filter(v => v !== null && v !== '');
    const embarkedMode = fitScaler ? getMode(embarkedValues) : scaler.embarkedMode;
    if (fitScaler) scaler.embarkedMode = embarkedMode;
    
    // Fare - use median for missing values
    const fareValues = data.map(row => row.Fare).filter(v => v !== null && !isNaN(v));
    const fareMedian = fitScaler ? getMedian(fareValues) : scaler.fareMedian;
    if (fitScaler) scaler.fareMedian = fareMedian;
    
    // Apply imputation
    processed.forEach(row => {
        if (row.Age === null || isNaN(row.Age)) row.Age = ageMedian;
        if (!row.Embarked) row.Embarked = embarkedMode;
        if (row.Fare === null || isNaN(row.Fare)) row.Fare = fareMedian;
    });
    
    // Feature Engineering
    const addFamilySize = document.getElementById('family-size-feature').checked;
    const addIsAlone = document.getElementById('is-alone-feature').checked;
    
    processed.forEach(row => {
        if (addFamilySize) {
            row.FamilySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
        }
        if (addIsAlone) {
            const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
            row.IsAlone = familySize === 1 ? 1 : 0;
        }
    });
    
    // Standardization for numerical features
    if (fitScaler) {
        // Calculate mean and std for Age and Fare
        const ages = processed.map(row => row.Age);
        const fares = processed.map(row => row.Fare);
        
        scaler.ageMean = ages.reduce((a, b) => a + b, 0) / ages.length;
        scaler.ageStd = Math.sqrt(ages.reduce((sum, val) => sum + Math.pow(val - scaler.ageMean, 2), 0) / ages.length);
        
        scaler.fareMean = fares.reduce((a, b) => a + b, 0) / fares.length;
        scaler.fareStd = Math.sqrt(fares.reduce((sum, val) => sum + Math.pow(val - scaler.fareMean, 2), 0) / fares.length);
    }
    
    // Apply standardization
    processed.forEach(row => {
        row.Age_scaled = (row.Age - scaler.ageMean) / (scaler.ageStd || 1);
        row.Fare_scaled = (row.Fare - sc
