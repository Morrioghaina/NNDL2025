<script type="module">
    import DataLoader from './data-loader.js';
    import CNNModel from './cnn.js';

    class ChickenDiseaseApp {
        constructor() {
            this.dataLoader = new DataLoader();
            this.model = null;
            this.currentData = null;
            this.trainingHistory = { loss: [], accuracy: [], valLoss: [], valAccuracy: [] };
            this.initializeEventListeners();
        }

        initializeEventListeners() {
            const zipInput = document.getElementById('zipFile');
            const csvInput = document.getElementById('csvFile');
            const trainBtn = document.getElementById('trainBtn');

            const checkFiles = () => {
                trainBtn.disabled = !(zipInput.files[0] && csvInput.files[0]);
            };

            zipInput.addEventListener('change', checkFiles);
            csvInput.addEventListener('change', checkFiles);
            
            trainBtn.addEventListener('click', () => this.startTraining());
        }

        async startTraining() {
            const zipFile = document.getElementById('zipFile').files[0];
            const csvFile = document.getElementById('csvFile').files[0];
            
            if (!zipFile || !csvFile) {
                this.showMessage('Please select both ZIP and CSV files', 'error');
                return;
            }

            try {
                this.showMessage('Loading dataset and performing EDA...', 'info');
                document.getElementById('trainingSection').style.display = 'block';
                document.getElementById('trainBtn').disabled = true;

                // Load data - FIXED: using loadDataset instead of loadFromCSV
                this.currentData = await this.dataLoader.loadDataset(zipFile, csvFile);
                
                this.showMessage(`Dataset loaded successfully! Found ${this.currentData.labelMap.size} disease classes`, 'success');
                
                // Perform EDA
                this.performEDA();
                
                // Create and train model
                this.model = new CNNModel(this.currentData.labelMap.size);
                this.model.createModel();
                
                await this.model.train(
                    this.currentData.X_train, 
                    this.currentData.y_train,
                    this.currentData.X_test,
                    this.currentData.y_test,
                    20,
                    {
                        onEpochEnd: (epoch, logs) => {
                            this.updateTrainingProgress(epoch, logs);
                        }
                    }
                );

                // Evaluate model and show comprehensive results
                const metrics = await this.model.computeMetrics(
                    this.currentData.X_test, 
                    this.currentData.y_test,
                    this.currentData.reverseLabelMap
                );
                
                this.displayComprehensiveResults(metrics);
                document.getElementById('resultsSection').style.display = 'block';

            } catch (error) {
                this.showMessage('Error: ' + error.message, 'error');
                console.error(error);
            }
        }

        performEDA() {
            this.showDatasetInfo();
            this.createClassDistributionChart();
            this.createTrainTestSplitChart();
        }

        showDatasetInfo() {
            const infoDiv = document.getElementById('datasetInfo');
            const totalSamples = this.currentData.originalImages.length;
            const trainSamples = this.currentData.trainIndices.length;
            const testSamples = this.currentData.testIndices.length;
            
            // Calculate class distribution
            const classDistribution = {};
            this.currentData.originalImages.forEach(img => {
                classDistribution[img.label] = (classDistribution[img.label] || 0) + 1;
            });

            infoDiv.innerHTML = `
                <h3>Dataset Overview</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Samples</td><td>${totalSamples}</td></tr>
                    <tr><td>Training Samples</td><td>${trainSamples} (${((trainSamples/totalSamples)*100).toFixed(1)}%)</td></tr>
                    <tr><td>Test Samples</td><td>${testSamples} (${((testSamples/totalSamples)*100).toFixed(1)}%)</td></tr>
                    <tr><td>Number of Classes</td><td>${this.currentData.labelMap.size}</td></tr>
                    <tr><td>Classes</td><td>${Array.from(this.currentData.labelMap.keys()).join(', ')}</td></tr>
                </table>
            `;
        }

        createClassDistributionChart() {
            const classDistribution = {};
            this.currentData.originalImages.forEach(img => {
                classDistribution[img.label] = (classDistribution[img.label] || 0) + 1;
            });

            const ctx = document.getElementById('classDistributionChart').getContext('2d');
            new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: Object.keys(classDistribution),
                    datasets: [{
                        data: Object.values(classDistribution),
                        backgroundColor: [
                            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', 
                            '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Class Distribution'
                        },
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        }

        createTrainTestSplitChart() {
            const total = this.currentData.originalImages.length;
            const train = this.currentData.trainIndices.length;
            const test = this.currentData.testIndices.length;

            const ctx = document.getElementById('trainTestSplitChart').getContext('2d');
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Training Set', 'Test Set'],
                    datasets: [{
                        data: [train, test],
                        backgroundColor: ['#4CAF50', '#2196F3']
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Train-Test Split'
                        }
                    }
                }
            });
        }

        updateTrainingProgress(epoch, logs) {
            document.getElementById('currentEpoch').textContent = epoch + 1;
            document.getElementById('currentLoss').textContent = logs.loss.toFixed(4);
            document.getElementById('currentAccuracy').textContent = (logs.acc * 100).toFixed(2);
            document.getElementById('currentValLoss').textContent = logs.val_loss ? logs.val_loss.toFixed(4) : '-';
            document.getElementById('currentValAccuracy').textContent = logs.val_acc ? (logs.val_acc * 100).toFixed(2) : '-';
            document.getElementById('epochProgress').style.width = `${((epoch + 1) / 20) * 100}%`;

            // Store history for charts
            this.trainingHistory.loss.push(logs.loss);
            this.trainingHistory.accuracy.push(logs.acc);
            if (logs.val_loss) this.trainingHistory.valLoss.push(logs.val_loss);
            if (logs.val_acc) this.trainingHistory.valAccuracy.push(logs.val_acc);
        }

        displayComprehensiveResults(metrics) {
            this.showModelMetrics(metrics);
            this.createTrainingCharts();
            this.showDiseaseAnalysis(metrics);
            this.showConfusionMatrix(metrics);
            this.showSamplePredictions(metrics);
        }

        showModelMetrics(metrics) {
            // Calculate overall metrics
            const totalCorrect = metrics.trueLabels.reduce((acc, trueLabel, idx) => 
                acc + (trueLabel === metrics.predictions[idx] ? 1 : 0), 0
            );
            const overallAccuracy = totalCorrect / metrics.trueLabels.length;

            // For multi-class, we can use micro-averaging for precision/recall
            const precision = overallAccuracy; // Simplified
            const recall = overallAccuracy;    // Simplified
            const f1Score = 2 * (precision * recall) / (precision + recall);

            document.getElementById('overallAccuracy').textContent = (overallAccuracy * 100).toFixed(1) + '%';
            document.getElementById('precision').textContent = (precision * 100).toFixed(1) + '%';
            document.getElementById('recall').textContent = (recall * 100).toFixed(1) + '%';
            document.getElementById('f1Score').textContent = (f1Score * 100).toFixed(1) + '%';
        }

        createTrainingCharts() {
            // Loss Chart
            const lossCtx = document.getElementById('lossChart').getContext('2d');
            new Chart(lossCtx, {
                type: 'line',
                data: {
                    labels: this.trainingHistory.loss.map((_, i) => i + 1),
                    datasets: [
                        {
                            label: 'Training Loss',
                            data: this.trainingHistory.loss,
                            borderColor: '#FF6384',
                            backgroundColor: 'rgba(255, 99, 132, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'Validation Loss',
                            data: this.trainingHistory.valLoss,
                            borderColor: '#36A2EB',
                            backgroundColor: 'rgba(54, 162, 235, 0.1)',
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: { display: true, text: 'Training & Validation Loss' }
                    }
                }
            });

            // Accuracy Chart
            const accCtx = document.getElementById('accuracyChart').getContext('2d');
            new Chart(accCtx, {
                type: 'line',
                data: {
                    labels: this.trainingHistory.accuracy.map((_, i) => i + 1),
                    datasets: [
                        {
                            label: 'Training Accuracy',
                            data: this.trainingHistory.accuracy.map(acc => acc * 100),
                            borderColor: '#4CAF50',
                            backgroundColor: 'rgba(76, 175, 80, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'Validation Accuracy',
                            data: this.trainingHistory.valAccuracy.map(acc => acc * 100),
                            borderColor: '#FF9800',
                            backgroundColor: 'rgba(255, 152, 0, 0.1)',
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: { display: true, text: 'Training & Validation Accuracy (%)' }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }

        showDiseaseAnalysis(metrics) {
            const rankingDiv = document.getElementById('diseaseRanking');
            rankingDiv.innerHTML = '<h3>Disease Classification Performance</h3>';
            
            const table = document.createElement('table');
            table.innerHTML = `
                <thead>
                    <tr><th>Rank</th><th>Disease</th><th>Accuracy</th><th>Test Samples</th><th>Status</th></tr>
                </thead>
                <tbody>
                    ${metrics.diseaseRanking.map((item, index) => {
                        const status = item.accuracy >= 0.8 ? '✅ Excellent' : 
                                     item.accuracy >= 0.6 ? '⚠️ Good' : 
                                     item.accuracy >= 0.4 ? '🔶 Fair' : '❌ Poor';
                        return `
                            <tr>
                                <td>${index + 1}</td>
                                <td>${item.disease}</td>
                                <td>${(item.accuracy * 100).toFixed(1)}%</td>
                                <td>${item.samples}</td>
                                <td>${status}</td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            `;
            rankingDiv.appendChild(table);

            // Create accuracy by disease chart
            this.createAccuracyByDiseaseChart(metrics.diseaseRanking);
        }

        createAccuracyByDiseaseChart(ranking) {
            const ctx = document.getElementById('accuracyByDiseaseChart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ranking.map(item => item.disease),
                    datasets: [{
                        label: 'Accuracy (%)',
                        data: ranking.map(item => item.accuracy * 100),
                        backgroundColor: ranking.map(item => 
                            item.accuracy >= 0.8 ? '#4CAF50' :
                            item.accuracy >= 0.6 ? '#FF9800' :
                            item.accuracy >= 0.4 ? '#FFC107' : '#f44336'
                        )
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: { display: true, text: 'Accuracy by Disease Class' }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: { display: true, text: 'Accuracy (%)' }
                        }
                    }
                }
            });
        }

        showConfusionMatrix(metrics) {
            const matrixDiv = document.getElementById('confusionMatrix');
            const diseases = metrics.diseaseRanking.map(item => item.disease);
            
            let html = '<table class="confusion-table"><tr><th></th>';
            
            // Header row
            diseases.forEach(disease => {
                html += `<th>${disease}</th>`;
            });
            html += '</tr>';
            
            // Data rows
            metrics.confusionMatrix.forEach((row, i) => {
                html += `<tr><th>${diseases[i]}</th>`;
                row.forEach((cell, j) => {
                    const maxVal = Math.max(...row);
                    const intensity = cell / maxVal;
                    html += `<td style="background-color: rgba(255,0,0,${intensity}); color: ${intensity > 0.5 ? 'white' : 'black'}">${cell}</td>`;
                });
                html += '</tr>';
            });
            html += '</table>';
            
            matrixDiv.innerHTML = html;
        }

        showSamplePredictions(metrics) {
            const samplesDiv = document.getElementById('samplePredictions');
            const container = document.createElement('div');
            container.className = 'sample-predictions';
            
            const numSamples = Math.min(12, metrics.trueLabels.length);
            
            for (let i = 0; i < numSamples; i++) {
                const trueLabel = metrics.trueLabels[i];
                const predLabel = metrics.predictions[i];
                const isCorrect = trueLabel === predLabel;
                const confidence = metrics.predictionProbabilities ? metrics.predictionProbabilities[i] : 'N/A';
                
                const sampleDiv = document.createElement('div');
                sampleDiv.className = `sample ${isCorrect ? 'correct' : 'incorrect'}`;
                sampleDiv.innerHTML = `
                    <div class="sample-image">Sample ${i + 1}</div>
                    <div class="sample-info">
                        <div><strong>True:</strong> ${this.currentData.reverseLabelMap.get(trueLabel)}</div>
                        <div><strong>Predicted:</strong> ${this.currentData.reverseLabelMap.get(predLabel)}</div>
                        <div style="font-size: 11px; color: #666;">Confidence: ${typeof confidence === 'number' ? (confidence * 100).toFixed(1) + '%' : confidence}</div>
                        <div style="color: ${isCorrect ? '#4CAF50' : '#f44336'}; font-weight: bold; margin-top: 5px;">
                            ${isCorrect ? '✓ Correct' : '✗ Incorrect'}
                        </div>
                    </div>
                `;
                
                container.appendChild(sampleDiv);
            }
            
            samplesDiv.innerHTML = '<h3>Test Sample Predictions</h3>';
            samplesDiv.appendChild(container);
        }

        showMessage(message, type = 'info') {
            const messageDiv = document.getElementById('message');
            messageDiv.textContent = message;
            messageDiv.className = `message ${type}`;
        }
    }

    // Initialize app
    document.addEventListener('DOMContentLoaded', () => {
        new ChickenDiseaseApp();
    });
</script>
