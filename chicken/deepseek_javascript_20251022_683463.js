import DataLoader from './data-loader.js';
import CNNModel from './cnn.js';

class ChickenDiseaseApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = null;
        this.currentData = null;
        this.isTraining = false;
        
        this.initializeEventListeners();
        this.updateUIState('initial');
    }

    initializeEventListeners() {
        // File upload handlers
        document.getElementById('zipFile').addEventListener('change', (e) => this.handleFileUpload(e));
        document.getElementById('csvFile').addEventListener('change', (e) => this.handleFileUpload(e));
        
        // Training handler
        document.getElementById('trainBtn').addEventListener('click', () => this.startTraining());
        
        // Model load handler
        document.getElementById('loadModelBtn').addEventListener('click', () => this.loadSavedModel());
    }

    handleFileUpload(event) {
        const files = {
            zip: document.getElementById('zipFile').files[0],
            csv: document.getElementById('csvFile').files[0]
        };
        
        if (files.zip && files.csv) {
            document.getElementById('trainBtn').disabled = false;
        }
    }

    async loadSavedModel() {
        try {
            this.showMessage('Loading saved model...', 'info');
            
            if (!this.model) {
                // We need to know numClasses, so we still need data
                if (!this.currentData) {
                    this.showMessage('Please load dataset first to determine number of classes', 'error');
                    return;
                }
                this.model = new CNNModel(this.currentData.labelMap.size);
            }
            
            const success = await this.model.loadModel();
            if (success) {
                this.showMessage('Model loaded successfully!', 'success');
                this.updateUIState('trained');
            } else {
                this.showMessage('No saved model found', 'error');
            }
        } catch (error) {
            this.showMessage('Error loading model: ' + error.message, 'error');
        }
    }

    async startTraining() {
        if (this.isTraining) return;
        
        const zipFile = document.getElementById('zipFile').files[0];
        const csvFile = document.getElementById('csvFile').files[0];
        
        if (!zipFile || !csvFile) {
            this.showMessage('Please select both ZIP and CSV files', 'error');
            return;
        }

        try {
            this.isTraining = true;
            this.updateUIState('training');
            this.showMessage('Loading dataset...', 'info');

            // Load and preprocess data
            this.currentData = await this.dataLoader.loadDataset(zipFile, csvFile);
            
            this.showMessage(`Dataset loaded: ${this.currentData.labelMap.size} disease classes`, 'success');
            
            // Create and train model
            this.model = new CNNModel(this.currentData.labelMap.size);
            this.model.createModel();
            
            this.showMessage('Starting training...', 'info');
            
            await this.model.train(
                this.currentData.X_train, 
                this.currentData.y_train,
                this.currentData.X_test,
                this.currentData.y_test,
                20, // epochs
                {
                    onEpochEnd: (epoch, logs) => {
                        this.updateTrainingProgress(epoch, logs);
                    },
                    onTrainEnd: () => {
                        this.onTrainingComplete();
                    }
                }
            );

        } catch (error) {
            this.showMessage('Training failed: ' + error.message, 'error');
            console.error(error);
            this.isTraining = false;
            this.updateUIState('initial');
        }
    }

    updateTrainingProgress(epoch, logs) {
        const progressDiv = document.getElementById('trainingProgress');
        const lossElem = document.getElementById('currentLoss');
        const accuracyElem = document.getElementById('currentAccuracy');
        
        if (lossElem) lossElem.textContent = logs.loss.toFixed(4);
        if (accuracyElem) accuracyElem.textContent = (logs.acc * 100).toFixed(2);
        
        // Update progress bar
        const progressBar = document.getElementById('epochProgress');
        if (progressBar) {
            progressBar.style.width = `${((epoch + 1) / 20) * 100}%`;
        }
    }

    async onTrainingComplete() {
        this.isTraining = false;
        this.showMessage('Training completed! Evaluating model...', 'success');
        
        try {
            // Save model
            await this.model.saveModel();
            
            // Evaluate model
            const metrics = await this.model.computeMetrics(
                this.currentData.X_test, 
                this.currentData.y_test,
                this.currentData.reverseLabelMap
            );
            
            this.displayResults(metrics);
            this.updateUIState('trained');
            
        } catch (error) {
            this.showMessage('Evaluation failed: ' + error.message, 'error');
        }
    }

    displayResults(metrics) {
        this.showDiseaseRanking(metrics.diseaseRanking);
        this.showConfusionMatrix(metrics.confusionMatrix, metrics.diseaseRanking);
        this.showSamplePredictions(metrics);
    }

    showDiseaseRanking(ranking) {
        const rankingDiv = document.getElementById('diseaseRanking');
        rankingDiv.innerHTML = '<h3>Disease Classification Accuracy Ranking</h3>';
        
        const table = document.createElement('table');
        table.className = 'ranking-table';
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Disease</th>
                    <th>Accuracy</th>
                    <th>Samples</th>
                </tr>
            </thead>
            <tbody>
                ${ranking.map((item, index) => `
                    <tr>
                        <td>${index + 1}</td>
                        <td>${item.disease}</td>
                        <td>${(item.accuracy * 100).toFixed(1)}%</td>
                        <td>${item.samples}</td>
                    </tr>
                `).join('')}
            </tbody>
        `;
        
        rankingDiv.appendChild(table);
    }

    showConfusionMatrix(matrix, ranking) {
        const matrixDiv = document.getElementById('confusionMatrix');
        matrixDiv.innerHTML = '<h3>Confusion Matrix</h3>';
        
        const table = document.createElement('table');
        table.className = 'confusion-matrix';
        
        // Create header row
        let headerRow = '<tr><th></th>';
        ranking.forEach(item => {
            headerRow += `<th>${item.disease}</th>`;
        });
        headerRow += '</tr>';
        
        // Create data rows
        let rows = '';
        matrix.forEach((row, i) => {
            const diseaseName = this.currentData.reverseLabelMap.get(i);
            rows += `<tr><th>${diseaseName}</th>`;
            row.forEach(cell => {
                const intensity = Math.min(100, (cell / Math.max(...row)) * 100);
                rows += `<td style="background-color: rgba(255,0,0,${intensity/100})">${cell}</td>`;
            });
            rows += '</tr>';
        });
        
        table.innerHTML = headerRow + rows;
        matrixDiv.appendChild(table);
    }

    showSamplePredictions(metrics) {
        const samplesDiv = document.getElementById('samplePredictions');
        samplesDiv.innerHTML = '<h3>Sample Predictions</h3>';
        
        const container = document.createElement('div');
        container.className = 'samples-container';
        
        // Show first 10 test samples
        const numSamples = Math.min(10, metrics.trueLabels.length);
        
        for (let i = 0; i < numSamples; i++) {
            const trueLabel = metrics.trueLabels[i];
            const predLabel = metrics.predictions[i];
            const isCorrect = trueLabel === predLabel;
            
            const sampleDiv = document.createElement('div');
            sampleDiv.className = `sample ${isCorrect ? 'correct' : 'incorrect'}`;
            sampleDiv.innerHTML = `
                <div class="sample-image">Image ${i + 1}</div>
                <div class="sample-info">
                    <div>True: ${this.currentData.reverseLabelMap.get(trueLabel)}</div>
                    <div>Predicted: ${this.currentData.reverseLabelMap.get(predLabel)}</div>
                    <div class="status ${isCorrect ? 'correct' : 'incorrect'}">
                        ${isCorrect ? '✓ Correct' : '✗ Incorrect'}
                    </div>
                </div>
            `;
            
            container.appendChild(sampleDiv);
        }
        
        samplesDiv.appendChild(container);
    }

    showMessage(message, type = 'info') {
        const messageDiv = document.getElementById('message');
        messageDiv.textContent = message;
        messageDiv.className = `message ${type}`;
        messageDiv.style.display = 'block';
        
        if (type !== 'error') {
            setTimeout(() => {
                messageDiv.style.display = 'none';
            }, 3000);
        }
    }

    updateUIState(state) {
        const states = {
            initial: { trainBtn: false, loadBtn: true, training: false, results: false },
            training: { trainBtn: false, loadBtn: false, training: true, results: false },
            trained: { trainBtn: true, loadBtn: true, training: false, results: true }
        };
        
        const config = states[state];
        
        document.getElementById('trainBtn').disabled = !config.trainBtn;
        document.getElementById('loadModelBtn').disabled = !config.loadBtn;
        document.getElementById('trainingSection').style.display = config.training ? 'block' : 'none';
        document.getElementById('resultsSection').style.display = config.results ? 'block' : 'none';
    }

    dispose() {
        if (this.dataLoader) {
            this.dataLoader.dispose();
        }
        if (this.model) {
            this.model.dispose();
        }
        tf.disposeVariables();
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ChickenDiseaseApp();
});